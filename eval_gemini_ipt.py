"""
Gemini IPT (Iterative Prompt Tuning) Main Loop

This module implements the main MMPO algorithm loop for Gemini,
matching the Qwen implementation in ipt/run_bench_singleclass_IPT.py
"""

import os
import json
import random
import logging
import concurrent.futures
import threading
from io import BytesIO
from PIL import Image
from tqdm import tqdm

from google import genai
from google.genai import types
from google.genai.types import Content, Part

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils.gemini_eval_utils import (
    inference_with_retry,
    parse_json,
    REQUEST_LIMIT,
    MAX_WORKERS,
    get_request_stats,
    reset_request_stats,
    log_request_stats,
)
from utils.shared_eval_utils import RateLimiter

from utils.gemini_ipt_utils import (
    draw_colored_bboxes_on_image,
    extract_class_definition,
    pil_image_to_part,
    generate_initial_class_definition,
    refine_definition_with_fp,
    refine_definition_with_fn,
    get_gt_examples_for_class,
    get_negative_example_for_class,
    identify_worst_examples,
    compute_map_for_class,
    save_ipt_checkpoint,
    load_ipt_checkpoint,
)


# =============================================================================
# Run Detection on Single Image with Definition
# =============================================================================

def run_detection_on_image(client, model_id, image_path, class_name, class_definition,
                           categories_dict, rate_limiter, logger):
    """
    Run object detection on a single image using the given class definition.

    Args:
        client: Gemini client
        model_id: Model ID
        image_path: Path to image
        class_name: Class to detect
        class_definition: Text definition of the class
        categories_dict: Dict mapping category names to IDs
        rate_limiter: Rate limiter
        logger: Logger

    Returns:
        List of detection dicts with 'bbox' (COCO format) and 'score'
    """
    # Load image
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    img = Image.open(BytesIO(img_bytes))
    if img.mode == 'RGBA':
        img = img.convert('RGB')

    original_width, original_height = img.size

    # Save to bytes for API
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_bytes_for_api = img_byte_arr.getvalue()

    image_part = Part.from_bytes(data=img_bytes_for_api, mime_type="image/jpeg")

    # Build detection prompt with class definition
    prompt = f"""Detect the 2d bounding boxes of the following object: {class_name}

Class definition: {class_definition}

The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000.

Return bounding boxes as a JSON array with labels. Never return masks."""

    contents = [Content(role="user", parts=[image_part, Part(text=prompt)])]

    response_text, err = inference_with_retry(
        contents=contents,
        system_prompt="Return bounding boxes as a JSON array with labels. Never return masks.",
        model_id=model_id,
        logger=logger,
        rate_limiter=rate_limiter,
        client=client
    )

    if err:
        logger.warning(f"Detection failed for {image_path}: {err}")
        return []

    if not response_text:
        return []

    # Parse response
    boxes = parse_json(response_text, logger)
    if not boxes:
        return []

    # Convert to COCO format detections
    detections = []
    for box in boxes:
        if not isinstance(box, dict):
            continue

        bbox_2d = box.get("box_2d", box.get("bbox", box.get("bounding_box", None)))
        if not bbox_2d or len(bbox_2d) != 4:
            continue

        label = box.get("label", box.get("class", box.get("name", "unknown")))

        # Check if label matches our class
        if class_name.lower() not in label.lower():
            continue

        # Convert Gemini format [ymin, xmin, ymax, xmax] normalized to COCO [x, y, w, h]
        y1, x1, y2, x2 = bbox_2d

        abs_y1 = int(y1 * original_height / 1000)
        abs_x1 = int(x1 * original_width / 1000)
        abs_y2 = int(y2 * original_height / 1000)
        abs_x2 = int(x2 * original_width / 1000)

        # Ensure correct ordering
        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        width = abs_x2 - abs_x1
        height = abs_y2 - abs_y1

        detections.append({
            'bbox': [abs_x1, abs_y1, width, height],
            'score': 1.0  # Gemini 3 doesn't support logprobs
        })

    return detections


# =============================================================================
# Evaluate with Definition on All Training Images
# =============================================================================

def _process_single_image_for_eval(args):
    """Worker function for parallel image evaluation."""
    (img_id, img_info, train_dir, class_name, class_definition,
     categories_dict, cat_id, client, model_id, rate_limiter, logger, coco_gt) = args

    image_path = os.path.join(train_dir, img_info['file_name'])

    # Get GT annotations for this image
    ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
    gt_anns = coco_gt.loadAnns(ann_ids)

    # Run detection
    detections = run_detection_on_image(
        client, model_id, image_path, class_name, class_definition,
        categories_dict, rate_limiter, logger
    )

    # Build result dict
    result = {
        'img_id': img_id,
        'image_path': image_path,
        'gt_anns': gt_anns,
        'predictions': detections
    }

    # Build COCO format predictions for this image
    coco_preds = []
    for det in detections:
        coco_preds.append({
            'image_id': img_id,
            'category_id': cat_id,
            'bbox': det['bbox'],
            'score': det['score'],
            'area': det['bbox'][2] * det['bbox'][3],
            'iscrowd': 0
        })

    return result, coco_preds


def evaluate_with_definition(client, model_id, train_dir, coco_gt, cat_id, class_name,
                             class_definition, categories_dict, rate_limiter, logger):
    """
    Evaluate the current class definition on ALL training images (parallelized).

    Args:
        client: Gemini client
        model_id: Model ID
        train_dir: Path to training folder
        coco_gt: COCO ground truth object
        cat_id: Category ID
        class_name: Class name
        class_definition: Current class definition text
        categories_dict: Dict mapping category IDs to names
        rate_limiter: Rate limiter
        logger: Logger

    Returns:
        tuple: (all_results list, mAP float)

        all_results is a list of dicts with keys:
        - 'img_id': Image ID
        - 'image_path': Path to image
        - 'gt_anns': List of GT annotations
        - 'predictions': List of prediction dicts
    """
    all_img_ids = coco_gt.getImgIds()

    logger.info(f"Evaluating '{class_name}' on {len(all_img_ids)} training images (parallel, {MAX_WORKERS} workers)")

    # Prepare args for each image
    args_list = []
    for img_id in all_img_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        args_list.append((
            img_id, img_info, train_dir, class_name, class_definition,
            categories_dict, cat_id, client, model_id, rate_limiter, logger, coco_gt
        ))

    all_results = []
    all_predictions_coco = []

    # Process in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_process_single_image_for_eval, args): args[0] for args in args_list}

        with tqdm(total=len(futures), desc=f"Evaluating {class_name}", leave=False) as pbar:
            for future in concurrent.futures.as_completed(futures):
                img_id = futures[future]
                try:
                    result, coco_preds = future.result()
                    all_results.append(result)
                    all_predictions_coco.extend(coco_preds)
                except Exception as e:
                    logger.warning(f"Error processing image {img_id}: {e}")
                pbar.update(1)

    # Assign unique IDs to predictions (must be done after collecting all)
    for idx, pred in enumerate(all_predictions_coco):
        pred['id'] = idx

    # Compute mAP
    current_map = compute_map_for_class(all_predictions_coco, coco_gt, cat_id)

    return all_results, current_map


# =============================================================================
# Main IPT Loop (matches ipt/run_bench_singleclass_IPT.py iterative_prompt_refinement)
# =============================================================================

def run_ipt_for_dataset(client, model_id, dataset_dir, num_iterations=3,
                        checkpoint_dir="gemini_ipt_test_exp", rate_limiter=None, logger=None):
    """
    Main IPT loop for a dataset.

    Algorithm (matches Qwen implementation):
    1. Load train annotations, get class names
    2. For each class (sequentially):
        a. Check for existing checkpoint, resume if found
        b. If not resuming:
            - Stage 1a: Generate initial definition from ALL GT examples (green boxes)
            - Stage 1b: Contrastive refinement against EACH negative class
        c. For i in range(num_iterations):
            - Evaluate current definition on ALL training images
            - Compute mAP using COCOeval
            - If mAP regressed from previous iteration, revert to prev definition
            - Identify worst FP/FN examples (excluding previously used ones)
            - Refine with FN first (include missed objects)
            - Refine with FP second (exclude false detections)
            - Track best mAP, save best definition
            - Save checkpoint after each iteration
        d. Save best definition for this class
    3. Return optimized definitions for all classes

    Args:
        client: Gemini client
        model_id: Model ID (e.g., "gemini-3-pro-preview")
        dataset_dir: Path to dataset directory
        num_iterations: Number of IPT iterations per class
        checkpoint_dir: Directory for checkpoints (default: "gemini_ipt")
        rate_limiter: Rate limiter for API calls
        logger: Logger instance

    Returns:
        dict: Mapping from class name to optimized definition
    """
    dataset_name = os.path.basename(dataset_dir)
    train_dir = os.path.join(dataset_dir, "train")

    # Setup checkpoint directory
    dataset_checkpoint_dir = os.path.join(checkpoint_dir, dataset_name)
    os.makedirs(dataset_checkpoint_dir, exist_ok=True)

    # Setup visualization directory for debug images
    viz_dir = os.path.join(dataset_checkpoint_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Load COCO annotations
    ann_path = os.path.join(train_dir, "_annotations.coco.json")
    coco_gt = COCO(ann_path)

    ds_cat_ids = coco_gt.getCatIds()
    cat_dict = {cat_id: coco_gt.cats[cat_id]["name"] for cat_id in ds_cat_ids}
    categories_dict = cat_dict  # For detection

    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Categories: {cat_dict}")
    logger.info(f"Checkpoint dir: {dataset_checkpoint_dir}")

    # Initialize statistics tracking
    ipt_stats = {
        "dataset": dataset_name,
        "num_classes": len(ds_cat_ids),
        "num_iterations": num_iterations,
        "classes": {},
        "totals": {
            "initial_def_success": 0,
            "initial_def_failed": 0,
            "contrastive_refinements": 0,
            "contrastive_extractions_success": 0,
            "contrastive_extractions_failed": 0,
            "fn_refinements": 0,
            "fn_extractions_success": 0,
            "fn_extractions_failed": 0,
            "fp_refinements": 0,
            "fp_extractions_success": 0,
            "fp_extractions_failed": 0,
            "detection_calls": 0,
            "detection_empty_responses": 0,
        }
    }

    # Check for completed classes (resume support)
    all_refined_path = os.path.join(dataset_checkpoint_dir,
                                     f"all_refined_class_instructions_{dataset_name}.json")
    refined_class_instructions = {}
    if os.path.exists(all_refined_path):
        with open(all_refined_path, "r") as f:
            refined_class_instructions = json.load(f)
        logger.info(f"Loaded {len(refined_class_instructions)} completed class definitions")

    # Process each class sequentially
    for cat_id in ds_cat_ids:
        class_name = cat_dict[cat_id]

        # Skip if already completed
        if class_name in refined_class_instructions:
            logger.info(f"Skipping class '{class_name}' - already completed")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing class '{class_name}' [{ds_cat_ids.index(cat_id)+1}/{len(ds_cat_ids)}]")
        logger.info(f"{'='*60}")

        # Initialize per-class stats
        ipt_stats["classes"][class_name] = {
            "initial_def_success": False,
            "contrastive_refinements": 0,
            "contrastive_extractions_success": 0,
            "contrastive_extractions_failed": 0,
            "iterations": {},
            "best_mAP": 0.0,
            "definition_updates": 0,
        }
        class_stats = ipt_stats["classes"][class_name]

        # --- Check for iteration-level checkpoint ---
        checkpoint = load_ipt_checkpoint(dataset_checkpoint_dir, dataset_name, class_name)

        if checkpoint:
            start_iteration = checkpoint["last_completed_iteration"] + 1
            current_def = checkpoint["current_instructions"]
            best_def = checkpoint["best_instructions"]
            best_map = checkpoint["best_mAP"]
            prev_mAP = checkpoint["prev_mAP"]
            prev_instructions = checkpoint["prev_instructions"]
            instruction_refinements = checkpoint.get("instruction_refinements", {})
            prev_worst_examples_map = checkpoint.get("prev_worst_examples_map", {})
            logger.info(f"Resuming from iteration {start_iteration}")
            logger.info(f"Current best mAP: {best_map:.4f}")
        else:
            start_iteration = 0
            instruction_refinements = {}
            prev_worst_examples_map = {}

            # --- Stage 1a: Initial definition from positive examples ---
            logger.info(f"Stage 1a: Generating initial definition from GT examples")
            gt_examples = get_gt_examples_for_class(coco_gt, train_dir, cat_id, class_name)
            logger.info(f"Found {len(gt_examples)} GT examples for '{class_name}'")

            current_def = generate_initial_class_definition(
                client, model_id, class_name, gt_examples, rate_limiter, logger,
                debug_dir=dataset_checkpoint_dir
            )

            # Track stats
            if current_def:
                class_stats["initial_def_success"] = True
                ipt_stats["totals"]["initial_def_success"] += 1
            else:
                ipt_stats["totals"]["initial_def_failed"] += 1
                logger.warning(f"Failed to generate initial definition for '{class_name}', skipping class")
                continue  # Skip to next class

            # Save initial definition
            init_def_path = os.path.join(dataset_checkpoint_dir, f"{class_name}_initial_definition.txt")
            with open(init_def_path, "w", encoding="utf-8") as f:
                f.write(current_def)
            logger.info(f"Saved initial definition to {init_def_path}")

            # --- Stage 1b: Contrastive refinement against each negative class ---
            logger.info(f"Stage 1b: Contrastive refinement against {len(ds_cat_ids)-1} negative classes")

            for idx, other_cat_id in enumerate(ds_cat_ids):
                if other_cat_id == cat_id:
                    continue

                other_class_name = cat_dict[other_cat_id]
                logger.info(f"  Contrasting against '{other_class_name}'")

                negative_example = get_negative_example_for_class(coco_gt, train_dir, other_cat_id)
                if negative_example is None:
                    logger.warning(f"  No negative example found for '{other_class_name}', skipping")
                    continue

                # Use a positive example (cycle through gt_examples)
                positive_example = gt_examples[idx % len(gt_examples)]

                refined_response = refine_definition_with_fp(
                    client, model_id, class_name, current_def,
                    positive_example, negative_example, rate_limiter, logger,
                    debug_dir=dataset_checkpoint_dir, debug_suffix=f"contrast_{other_class_name}"
                )

                # Track stats
                class_stats["contrastive_refinements"] += 1
                ipt_stats["totals"]["contrastive_refinements"] += 1

                # Handle None response (timeout/error)
                if refined_response is None:
                    class_stats["contrastive_extractions_failed"] += 1
                    ipt_stats["totals"]["contrastive_extractions_failed"] += 1
                    logger.warning(f"  Refinement call failed for contrast with '{other_class_name}', keeping current definition")
                    continue

                extracted = extract_class_definition(refined_response, class_name)
                if extracted:
                    current_def = extracted
                    class_stats["contrastive_extractions_success"] += 1
                    class_stats["definition_updates"] += 1
                    ipt_stats["totals"]["contrastive_extractions_success"] += 1
                    logger.info(f"  Updated definition after contrast with '{other_class_name}'")
                else:
                    class_stats["contrastive_extractions_failed"] += 1
                    ipt_stats["totals"]["contrastive_extractions_failed"] += 1
                    logger.warning(f"  Failed to extract definition after contrast with '{other_class_name}'")

                    # Save intermediate definition (raw response already saved by refine_definition_with_fp)
                    contrast_def_path = os.path.join(
                        dataset_checkpoint_dir,
                        f"{class_name}_initial_definition_contrast_{other_class_name}.txt"
                    )
                    with open(contrast_def_path, "w", encoding="utf-8") as f:
                        f.write(current_def)

            # Save final Stage 1 definition
            stage1_def_path = os.path.join(dataset_checkpoint_dir, f"{class_name}_stage1_definition.txt")
            with open(stage1_def_path, "w", encoding="utf-8") as f:
                f.write(current_def)
            logger.info(f"Completed Stage 1. Saved to {stage1_def_path}")

            best_def = current_def
            best_map = -1.0
            prev_mAP = -1.0
            prev_instructions = current_def

        # Skip iterations if all already done
        if start_iteration >= num_iterations:
            logger.info(f"All {num_iterations} iterations already completed for '{class_name}'")
            refined_class_instructions[class_name] = best_def
            continue

        # --- Stage 2: Iterative refinement ---
        logger.info(f"\nStage 2: Iterative refinement ({num_iterations} iterations)")

        # Note: prev_worst_examples_map is initialized above (either from checkpoint or empty dict)

        for i in range(start_iteration, num_iterations):
            logger.info(f"\n--- Iteration {i} for class '{class_name}' ---")

            # Evaluate on ALL training images
            all_results, current_map = evaluate_with_definition(
                client, model_id, train_dir, coco_gt, cat_id, class_name,
                current_def, categories_dict, rate_limiter, logger
            )

            logger.info(f"Iteration {i}: mAP = {current_map:.4f}")

            # Track best
            if current_map >= best_map:
                best_map = current_map
                best_def = current_def
                logger.info(f"  New best mAP: {best_map:.4f}")

            # Check for regression
            if prev_mAP > 0 and current_map < prev_mAP:
                logger.info(f"  mAP regressed ({prev_mAP:.4f} -> {current_map:.4f}), reverting to previous definition")
                current_def = prev_instructions
            else:
                prev_instructions = current_def
                prev_mAP = current_map

            # Find worst examples
            few_shot_examples, prev_worst_examples_map = identify_worst_examples(
                all_results, cat_id, class_name, coco_gt, train_dir,
                prev_worst_examples_map, viz_dir=viz_dir, iteration=i
            )

            # Log found examples with counts (worst_fp and worst_fn are now lists)
            fp_count = len(few_shot_examples.get('worst_fp', [])) if isinstance(few_shot_examples.get('worst_fp'), list) else (1 if 'worst_fp' in few_shot_examples else 0)
            fn_count = len(few_shot_examples.get('worst_fn', [])) if isinstance(few_shot_examples.get('worst_fn'), list) else (1 if 'worst_fn' in few_shot_examples else 0)
            logger.info(f"  Found examples - best_match: {'yes' if 'best_match' in few_shot_examples else 'no'}, "
                       f"worst_fp: {fp_count}, worst_fn: {fn_count}")

            # Initialize iteration stats
            iter_stats = {
                "mAP": current_map,
                "fn_refinement_success": False,
                "fp_refinement_success": False,
            }

            # Build other class definitions for diversification constraint (Mitigation 4)
            other_class_defs = {k: v for k, v in refined_class_instructions.items() if k != class_name}

            # Refine with FN first (per Qwen method_refine_prompt order)
            # Note: worst_fn is now a list of images (Mitigation 5: top 3 FN)
            if 'best_match' in few_shot_examples and 'worst_fn' in few_shot_examples:
                num_fn = len(few_shot_examples['worst_fn']) if isinstance(few_shot_examples['worst_fn'], list) else 1
                logger.info(f"  Refining with {num_fn} FN example(s)")
                ipt_stats["totals"]["fn_refinements"] += 1
                fn_refined = refine_definition_with_fn(
                    client, model_id, class_name, current_def,
                    few_shot_examples['best_match'], few_shot_examples['worst_fn'],
                    rate_limiter, logger,
                    debug_dir=dataset_checkpoint_dir, debug_suffix=f"iter{i}",
                    other_class_definitions=other_class_defs if other_class_defs else None
                )
                # Handle None response (timeout/error)
                if fn_refined is None:
                    ipt_stats["totals"]["fn_extractions_failed"] += 1
                    logger.warning(f"  FN refinement call failed, keeping current definition")
                else:
                    extracted = extract_class_definition(fn_refined, class_name)
                    if extracted:
                        current_def = extracted
                        iter_stats["fn_refinement_success"] = True
                        class_stats["definition_updates"] += 1
                        ipt_stats["totals"]["fn_extractions_success"] += 1
                        logger.info(f"  Updated definition after FN refinement")
                    else:
                        ipt_stats["totals"]["fn_extractions_failed"] += 1
                        logger.warning(f"  Failed to extract definition after FN refinement")

            # Refine with FP second
            # Note: worst_fp is now a list of images (Mitigation 5: top 3 FP)
            if 'best_match' in few_shot_examples and 'worst_fp' in few_shot_examples:
                num_fp = len(few_shot_examples['worst_fp']) if isinstance(few_shot_examples['worst_fp'], list) else 1
                logger.info(f"  Refining with {num_fp} FP example(s)")
                ipt_stats["totals"]["fp_refinements"] += 1
                fp_refined = refine_definition_with_fp(
                    client, model_id, class_name, current_def,
                    few_shot_examples['best_match'], few_shot_examples['worst_fp'],
                    rate_limiter, logger,
                    debug_dir=dataset_checkpoint_dir, debug_suffix=f"iter{i}",
                    other_class_definitions=other_class_defs if other_class_defs else None
                )
                # Handle None response (timeout/error)
                if fp_refined is None:
                    ipt_stats["totals"]["fp_extractions_failed"] += 1
                    logger.warning(f"  FP refinement call failed, keeping current definition")
                else:
                    extracted = extract_class_definition(fp_refined, class_name)
                    if extracted:
                        current_def = extracted
                        iter_stats["fp_refinement_success"] = True
                        class_stats["definition_updates"] += 1
                        ipt_stats["totals"]["fp_extractions_success"] += 1
                        logger.info(f"  Updated definition after FP refinement")
                    else:
                        ipt_stats["totals"]["fp_extractions_failed"] += 1
                        logger.warning(f"  Failed to extract definition after FP refinement")
                        # Raw response already saved by refine_definition_with_fp

            # Save iteration stats
            class_stats["iterations"][f"iter_{i}"] = iter_stats

            # Save checkpoint
            instruction_refinements[f"iter_{i}"] = {
                "mAP": current_map,
                "definition": current_def
            }
            save_ipt_checkpoint(
                dataset_checkpoint_dir, dataset_name, class_name, i,
                current_def, best_def, best_map, prev_mAP,
                prev_instructions, instruction_refinements, prev_worst_examples_map
            )
            logger.info(f"  Saved checkpoint for iteration {i}")

        # Save completed class
        logger.info(f"\nCompleted class '{class_name}' with best mAP = {best_map:.4f}")

        # Update class stats with final best mAP
        class_stats["best_mAP"] = best_map

        # Save best definition to file
        best_def_path = os.path.join(dataset_checkpoint_dir, f"{class_name}_best_definition.txt")
        with open(best_def_path, "w", encoding="utf-8") as f:
            f.write(best_def)
        logger.info(f"Saved best definition to {best_def_path}")

        # Update refined instructions
        refined_class_instructions[class_name] = best_def
        with open(all_refined_path, "w", encoding="utf-8") as f:
            json.dump(refined_class_instructions, f, indent=2)
        logger.info(f"Updated all_refined_class_instructions")

        # Save stats incrementally after each class
        stats_path = os.path.join(dataset_checkpoint_dir, f"ipt_stats_{dataset_name}.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(ipt_stats, f, indent=2)
        logger.info(f"Updated IPT statistics")

    logger.info(f"\n{'='*60}")
    logger.info(f"IPT completed for dataset '{dataset_name}'")
    logger.info(f"Optimized {len(refined_class_instructions)} classes")
    logger.info(f"{'='*60}")

    # Add API request stats to ipt_stats
    ipt_stats["api_request_stats"] = get_request_stats()

    # Save IPT statistics
    stats_path = os.path.join(dataset_checkpoint_dir, f"ipt_stats_{dataset_name}.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(ipt_stats, f, indent=2)
    logger.info(f"Saved IPT statistics to {stats_path}")

    # Log summary statistics
    totals = ipt_stats["totals"]
    logger.info(f"\n--- IPT Statistics Summary ---")
    logger.info(f"Initial definitions: {totals['initial_def_success']} success, {totals['initial_def_failed']} failed")
    logger.info(f"Contrastive refinements: {totals['contrastive_refinements']} total, "
                f"{totals['contrastive_extractions_success']} extracted, "
                f"{totals['contrastive_extractions_failed']} failed")
    logger.info(f"FN refinements: {totals['fn_refinements']} total, "
                f"{totals['fn_extractions_success']} extracted, "
                f"{totals['fn_extractions_failed']} failed")
    logger.info(f"FP refinements: {totals['fp_refinements']} total, "
                f"{totals['fp_extractions_success']} extracted, "
                f"{totals['fp_extractions_failed']} failed")

    # Log API request statistics
    log_request_stats(logger)

    return refined_class_instructions


# =============================================================================
# Main Entry Point (standalone execution)
# =============================================================================

def main():
    """Main function for standalone IPT execution."""
    import argparse

    parser = argparse.ArgumentParser(description='Run Gemini IPT for object detection')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Path to dataset directory (must contain train/ folder)')
    parser.add_argument('--model_id', type=str, default="gemini-3-pro-preview",
                        help='Gemini model ID')
    parser.add_argument('--num_iterations', type=int, default=3,
                        help='Number of IPT iterations per class')
    parser.add_argument('--checkpoint_dir', type=str, default="gemini_ipt_test_exp",
                        help='Directory for checkpoints')
    parser.add_argument('--api_key', type="AIzaSyDUPkTRcNjoRhGXAouaGIlGU02Zury8F10", default=None,
                        help='Gemini API key (or set GEMINI_API_KEY env var)')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"gemini_ipt_test_exp_{os.path.basename(args.dataset_dir)}.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Setup Gemini client with timeout
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY") or "AIzaSyDUPkTRcNjoRhGXAouaGIlGU02Zury8F10"
    # timeout is in milliseconds
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=120_000)
    )

    # Setup rate limiter
    rate_limiter = RateLimiter(REQUEST_LIMIT)

    # Run IPT
    optimized_definitions = run_ipt_for_dataset(
        client=client,
        model_id=args.model_id,
        dataset_dir=args.dataset_dir,
        num_iterations=args.num_iterations,
        checkpoint_dir=args.checkpoint_dir,
        rate_limiter=rate_limiter,
        logger=logger
    )

    logger.info(f"\nFinal optimized definitions:")
    for class_name, definition in optimized_definitions.items():
        logger.info(f"\n{class_name}:\n{definition[:200]}...")


if __name__ == "__main__":
    main()
