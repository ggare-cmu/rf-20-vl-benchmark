"""
Gemini IPT (Iterative Prompt Tuning) Utility Functions

This module implements the core functions for MMPO (Multi-Modal Prompt Optimization)
as described in paper2.tex and matching the Qwen implementation in ipt/run_bench_singleclass_IPT.py
"""

import os
import re
import ast
import json
import random
from io import BytesIO
from PIL import Image, ImageDraw
from google.genai.types import Content, Part

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.gemini_eval_utils import (
    inference_with_retry,
    parse_json,
    safety_settings,
)


# =============================================================================
# Bounding Box Drawing (matches ipt/ipt_utils.py lines 1227-1239)
# =============================================================================

def draw_colored_bboxes_on_image(image, color, bboxes):
    """
    Draws colored bboxes on the image.
    Returns a new PIL image with the boxes drawn.
    Bboxes should be [x, y, w, h] in COCO format (image coordinates).

    Args:
        image: PIL Image
        color: "green", "red", or "blue" (string color name)
        bboxes: List of [x, y, w, h] bounding boxes

    Returns:
        PIL Image with bboxes drawn (width=8 like Qwen implementation)
    """
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    for (x, y, w, h) in bboxes:
        draw.rectangle([(x, y), (x + w, y + h)], outline=color, width=2)

    return img_copy


# =============================================================================
# Extract Class Definition from Response (matches ipt/run_bench_singleclass_IPT.py lines 385-414)
# =============================================================================

def extract_class_definition(response, class_name):
    """
    Extract the class definition from model response.
    Handles ```python code blocks, dict parsing, and various formats.

    Args:
        response: Raw model response text
        class_name: The class name to extract definition for

    Returns:
        str: Extracted definition text, or None if parsing fails
    """
    if not response:
        return None

    # Find all code blocks (optionally labeled as python or json)
    code_blocks = re.findall(r"```(?:python|json)?\s*(.*?)\s*```", response, re.DOTALL)

    # If no code blocks found, try to parse the entire response as JSON/Python
    if not code_blocks:
        # Check if response looks like JSON (starts with [ or {)
        stripped = response.strip()
        if stripped.startswith('[') or stripped.startswith('{'):
            code_blocks = [stripped]

    for block in code_blocks:
        block = block.strip()
        parsed = None

        # Try parsing as a Python dict first (since many are not strict JSON)
        try:
            parsed = ast.literal_eval(block)
        except Exception:
            # If that fails, try JSON parsing
            try:
                parsed = json.loads(block)
            except Exception:
                continue

        # If we got a dict, try to extract the class definition
        if isinstance(parsed, dict):
            # Try exact match first
            if class_name in parsed:
                value = parsed[class_name]
                # Ensure we return a string
                if isinstance(value, str):
                    return value
                elif isinstance(value, dict):
                    # Try to extract a definition field from nested dict
                    for key in ['definition', 'generalizable_definition', 'description', 'text']:
                        if key in value:
                            return value[key]
                    # Fallback: stringify the dict
                    return json.dumps(value)
            # Try case-insensitive match
            for key, value in parsed.items():
                if key.lower() == class_name.lower():
                    if isinstance(value, str):
                        return value
                    elif isinstance(value, dict):
                        for k in ['definition', 'generalizable_definition', 'description', 'text']:
                            if k in value:
                                return value[k]
                        return json.dumps(value)

        # Handle list format: [{"class_name": "..."}] or [{"generalizable_definition": "..."}] or ["plain string"]
        elif isinstance(parsed, list) and len(parsed) > 0:
            first_item = parsed[0]
            # Handle ["plain string"] format
            if isinstance(first_item, str):
                return first_item
            elif isinstance(first_item, dict):
                # First check if the dict has the class_name as key: [{"action": "definition"}]
                if class_name in first_item:
                    value = first_item[class_name]
                    if isinstance(value, str):
                        return value
                # Also try case-insensitive match for class name
                for key, value in first_item.items():
                    if key.lower() == class_name.lower() and isinstance(value, str):
                        return value

                # Try to extract definition from common keys
                for key in ['generalizable_definition', 'definition', 'object_class_definition',
                           'description', 'text', 'subject_description', 'object_class_description']:
                    if key in first_item:
                        return first_item[key]
                # If no standard key found, try to build a definition from available fields
                if 'visual_characteristics' in first_item or 'distinctive_features' in first_item:
                    # This is a structured response - extract the most useful part
                    parts = []
                    if 'subject_description' in first_item:
                        parts.append(first_item['subject_description'])
                    if 'generalizable_definition' in first_item:
                        parts.append(first_item['generalizable_definition'])
                    if parts:
                        return ' '.join(parts)

    return None


# =============================================================================
# PIL Image to Gemini Part Conversion
# =============================================================================

def pil_image_to_part(image, max_dimension=(1920, 1080)):
    """
    Convert PIL image to Gemini Part, with optional resizing.

    Args:
        image: PIL Image
        max_dimension: Maximum (width, height) tuple for resizing

    Returns:
        google.genai.types.Part containing the image bytes
    """
    img_copy = image.copy()
    img_copy.thumbnail(max_dimension, Image.LANCZOS)

    # Convert to RGB if necessary
    if img_copy.mode == 'RGBA':
        img_copy = img_copy.convert('RGB')

    # Save to bytes
    img_byte_arr = BytesIO()
    img_copy.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()

    return Part.from_bytes(data=img_bytes, mime_type="image/jpeg")


# =============================================================================
# Initial Class Definition Generation (matches ipt/run_bench_singleclass_IPT.py lines 241-280)
# =============================================================================

def generate_initial_class_definition(client, model_id, class_name, few_shot_images_with_boxes,
                                       rate_limiter, logger, debug_dir=None):
    """
    Show GT examples with green bounding boxes, ask model to describe the class.
    This is Stage 1a of the MMPO algorithm.

    Args:
        client: Gemini client
        model_id: e.g., "gemini-3-pro-preview"
        class_name: e.g., "Adult"
        few_shot_images_with_boxes: List of PIL images with green boxes drawn on GT objects
        rate_limiter: Rate limiter for API calls
        logger: Logger instance
        debug_dir: Optional directory to save raw responses for debugging

    Returns:
        str: Initial class definition text
    """
    if not few_shot_images_with_boxes:
        raise ValueError(f"No few-shot examples provided for class '{class_name}'")

    # Prompt template from Qwen implementation (lines 252-261) with conciseness instructions
    prompt = f"""Analyze the following images and describe the subjects or objects highlighted in green bounding boxes.
Identify and summarize the key visual characteristics that are consistently observed across these objects.
Emphasize the distinctive features that clearly differentiate this object class from other elements in the scene.

Your goal is to produce a concise, clear, and detailed definition that enables accurate recognition of this object class. Keep your definition to 2-3 sentences maximum.

Do not mention bounding boxes, colors, or any annotation details in your response."""

    # Build content parts: text prompt followed by all images
    parts = [Part(text=prompt)]
    for img in few_shot_images_with_boxes:
        parts.append(pil_image_to_part(img))

    contents = [Content(role="user", parts=parts)]

    logger.info(f"Generating initial definition for '{class_name}' using {len(few_shot_images_with_boxes)} GT examples")

    response_text, err = inference_with_retry(
        contents=contents,
        system_prompt="You are a helpful assistant that describes visual characteristics of objects.",
        model_id=model_id,
        logger=logger,
        rate_limiter=rate_limiter,
        client=client,
        json_mode=False  # Definition generation returns plain text, not JSON
    )

    if err:
        logger.warning(f"Failed to generate initial definition for '{class_name}': {err}")
        return None

    # Save raw response for debugging
    if debug_dir:
        raw_path = os.path.join(debug_dir, f"{class_name}_initial_RAW_RESPONSE.txt")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(response_text if response_text else "None")

    if not response_text:
        logger.warning(f"Empty response for initial definition of '{class_name}'")
        return None

    # Clean up the definition
    definition = response_text.strip()
    definition = definition.replace(f"The visual characteristics of the '{class_name}' class are:", "").strip()
    definition = definition.replace(f"Definition of '{class_name}':", "").strip()

    # If the model returned JSON despite our instructions, try to extract plain text from it
    if definition.startswith('[') or definition.startswith('{') or '```' in definition:
        extracted = extract_class_definition(definition, class_name)
        if extracted:
            definition = extracted
            logger.info(f"Extracted definition from structured response for '{class_name}'")

    logger.info(f"Generated initial definition for '{class_name}': {definition[:200]}...")
    return definition


# =============================================================================
# FP-Based Refinement (matches ipt/run_bench_singleclass_IPT.py lines 284-331)
# Also used for Stage 1b contrastive refinement against negative classes
# =============================================================================

def refine_definition_with_fp(client, model_id, class_name, current_definition,
                               correct_image, fp_images, rate_limiter, logger,
                               debug_dir=None, debug_suffix="", other_class_definitions=None):
    """
    Show correct example (green box) vs false positives (red boxes), refine definition.
    Also used in Stage 1 for contrastive refinement against negative classes.

    Args:
        client: Gemini client
        model_id: Model ID
        class_name: Class being refined
        current_definition: Current class definition text
        correct_image: PIL image with green box on correct example
        fp_images: Single PIL image or list of PIL images with red boxes on false positives
        rate_limiter: Rate limiter
        logger: Logger
        debug_dir: Optional directory to save raw responses and prompts
        debug_suffix: Optional suffix for debug filename (e.g., "contrast_activity" or "iter0")
        other_class_definitions: Optional dict of {class_name: definition} for diversification

    Returns:
        str: Raw model response (needs extract_class_definition to parse)
    """
    # Handle both single image (backward compat) and list of images
    if not isinstance(fp_images, list):
        fp_images = [fp_images]

    num_fp = len(fp_images)
    fp_description = "the red bounding box" if num_fp == 1 else f"the {num_fp} red bounding boxes"

    # Build diversification constraint if other class definitions provided
    diversification_text = ""
    if other_class_definitions:
        diversification_text = f"""

IMPORTANT - Class Diversification Constraint:
Your definition for '{class_name}' must be meaningfully different from these other class definitions:
"""
        for other_class, other_def in other_class_definitions.items():
            diversification_text += f"- {other_class}: {other_def}\n"
        diversification_text += f"Ensure your definition captures unique characteristics that distinguish '{class_name}' from ALL other classes above."

    # Prompt template extended for multiple FPs and diversification
    prompt = f"""Analyze the images carefully and identify the key visual differences between the object shown in the green bounding box and the objects shown in {fp_description}.

Follow the following steps:
Step-1. Describe the distinguishing visual characteristics that set apart the object in the green bounding box from the objects in {fp_description}.
Step-2. Based on these distinguishing traits, formulate a clear and descriptive class definition for the object in the green bounding box. This definition should focus on its unique visual and contextual features that help differentiate it from the objects in {fp_description}.
Step-3. Compare your new class definition with the existing definition of the '{class_name}' class provided below:

Current class definition of the '{class_name}' class:
{current_definition}

Step-4. Synthesize both definitions to produce an improved, more precise descriptive class definition for the '{class_name}' class. The updated definition should make it easier to accurately identify true instances of the '{class_name}' class while reducing false positives similar to those seen in {fp_description}.
{diversification_text}
Important: Keep the definition concise (2-3 sentences maximum).

Note: Do not mention bounding boxes, colors, or image annotations in your response. The updated class definition should be a textual description of the '{class_name}' class objects.

Return the final updated class definition as descriptive text in the following format: ```python
{{'{class_name}': <updated definition>}}
```"""

    # Build content: text + correct image + FP images
    parts = [
        Part(text=prompt),
        pil_image_to_part(correct_image),
    ]
    for fp_img in fp_images:
        parts.append(pil_image_to_part(fp_img))

    contents = [Content(role="user", parts=parts)]

    logger.info(f"Refining definition for '{class_name}' with {num_fp} FP example(s)")

    response_text, err = inference_with_retry(
        contents=contents,
        system_prompt="You are a helpful assistant that refines object class definitions based on visual examples.",
        model_id=model_id,
        logger=logger,
        rate_limiter=rate_limiter,
        client=client,
        json_mode=False  # Definition refinement returns plain text, not JSON
    )

    if err:
        logger.warning(f"Failed to refine definition with FP for '{class_name}': {err}")
        return None

    # Save prompt and response for debugging
    if debug_dir:
        suffix = f"_{debug_suffix}" if debug_suffix else ""
        # Save prompt
        prompt_path = os.path.join(debug_dir, f"{class_name}_fp{suffix}_PROMPT.txt")
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt)
        # Save response
        raw_path = os.path.join(debug_dir, f"{class_name}_fp{suffix}_RAW_RESPONSE.txt")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(response_text if response_text else "None")

    logger.info(f"FP refinement response for '{class_name}': {response_text[:200] if response_text else 'None'}...")
    return response_text


# =============================================================================
# FN-Based Refinement (matches ipt/run_bench_singleclass_IPT.py lines 335-378)
# =============================================================================

def refine_definition_with_fn(client, model_id, class_name, current_definition,
                               correct_image, fn_images, rate_limiter, logger,
                               debug_dir=None, debug_suffix="", other_class_definitions=None):
    """
    Show correct example (green box) vs missed detections (blue boxes), expand definition.

    Args:
        client: Gemini client
        model_id: Model ID
        class_name: Class being refined
        current_definition: Current class definition text
        correct_image: PIL image with green box on correct example
        fn_images: Single PIL image or list of PIL images with blue boxes on missed detections
        rate_limiter: Rate limiter
        logger: Logger
        debug_dir: Optional directory to save raw responses and prompts
        debug_suffix: Optional suffix for debug filename (e.g., "iter0")
        other_class_definitions: Optional dict of {class_name: definition} for diversification

    Returns:
        str: Raw model response (needs extract_class_definition to parse)
    """
    # Handle both single image (backward compat) and list of images
    if not isinstance(fn_images, list):
        fn_images = [fn_images]

    num_fn = len(fn_images)
    fn_description = "the blue bounding box" if num_fn == 1 else f"the {num_fn} blue bounding boxes"

    # Build diversification constraint if other class definitions provided
    diversification_text = ""
    if other_class_definitions:
        diversification_text = f"""

IMPORTANT - Class Diversification Constraint:
Your definition for '{class_name}' must be meaningfully different from these other class definitions:
"""
        for other_class, other_def in other_class_definitions.items():
            diversification_text += f"- {other_class}: {other_def}\n"
        diversification_text += f"Ensure your definition captures unique characteristics that distinguish '{class_name}' from ALL other classes above."

    # Prompt template extended for multiple FNs and diversification
    prompt = f"""Analyze the images carefully and identify the key visual similarities between the object shown in the green bounding box and the objects shown in {fn_description}.

Follow the following steps:
Step-1. Describe the similar visual characteristics shared by the object in the green bounding box and the objects in {fn_description}.
Step-2. Based on these similarity traits, formulate a clear and descriptive class definition that covers both the object in the green bounding box and the objects in {fn_description}. This definition should focus on visual and contextual features that help identify all these instances.
Step-3. Compare your new class definition with the existing definition of the '{class_name}' class provided below:

Current class definition of the '{class_name}' class:
{current_definition}

Step-4. Synthesize both definitions to produce an improved, more precise descriptive class definition for the '{class_name}' class. The updated definition should make it easier to accurately identify all true instances of the '{class_name}' class, including variations like those seen in {fn_description}.
{diversification_text}
Important: Keep the definition concise (2-3 sentences maximum).

Note: Do not mention bounding boxes, colors, or image annotations in your response. The updated class definition should be a textual description of the '{class_name}' class objects.

Return the final updated class definition as descriptive text in the following format: ```python
{{'{class_name}': <updated definition>}}
```"""

    # Build content: text + correct image + FN images
    parts = [
        Part(text=prompt),
        pil_image_to_part(correct_image),
    ]
    for fn_img in fn_images:
        parts.append(pil_image_to_part(fn_img))

    contents = [Content(role="user", parts=parts)]

    logger.info(f"Refining definition for '{class_name}' with {num_fn} FN example(s)")

    response_text, err = inference_with_retry(
        contents=contents,
        system_prompt="You are a helpful assistant that refines object class definitions based on visual examples.",
        model_id=model_id,
        logger=logger,
        rate_limiter=rate_limiter,
        client=client,
        json_mode=False  # Definition refinement returns plain text, not JSON
    )

    if err:
        logger.warning(f"Failed to refine definition with FN for '{class_name}': {err}")
        return None

    # Save prompt and response for debugging
    if debug_dir:
        suffix = f"_{debug_suffix}" if debug_suffix else ""
        # Save prompt
        prompt_path = os.path.join(debug_dir, f"{class_name}_fn{suffix}_PROMPT.txt")
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt)
        # Save response
        raw_path = os.path.join(debug_dir, f"{class_name}_fn{suffix}_RAW_RESPONSE.txt")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(response_text if response_text else "None")

    logger.info(f"FN refinement response for '{class_name}': {response_text[:200] if response_text else 'None'}...")
    return response_text


# =============================================================================
# Get GT Examples for Class (matches ipt/run_bench_singleclass_IPT.py lines 425-475)
# =============================================================================

def get_gt_examples_for_class(coco_gt, train_dir, cat_id, class_name):
    """
    Get all GT examples for a class with green bounding boxes drawn.

    Args:
        coco_gt: COCO ground truth object
        train_dir: Path to training images
        cat_id: Category ID
        class_name: Class name (for logging)

    Returns:
        List of PIL images with green boxes drawn on GT objects
    """
    img_ids = coco_gt.getImgIds(catIds=[cat_id])
    if not img_ids:
        raise ValueError(f"No images found for category '{class_name}' (ID: {cat_id})")

    # Use all images (sorted for reproducibility)
    selected_img_ids = sorted(img_ids)

    gt_examples = []
    for img_id in selected_img_ids:
        ann_ids = coco_gt.getAnnIds(imgIds=[img_id], catIds=[cat_id])
        anns = coco_gt.loadAnns(ann_ids)

        img_info_list = coco_gt.loadImgs(img_id)
        if not img_info_list:
            continue

        img_info = img_info_list[0]
        image_path = os.path.join(train_dir, img_info["file_name"])
        if not os.path.isfile(image_path):
            continue

        # Get GT bboxes for this category only
        gt_bboxes = [ann['bbox'] for ann in anns if ann['category_id'] == cat_id]
        if not gt_bboxes:
            continue

        # Load image and draw green boxes
        img = Image.open(image_path).convert("RGB")
        img_with_boxes = draw_colored_bboxes_on_image(img, "green", gt_bboxes)

        gt_examples.append(img_with_boxes)

    if not gt_examples:
        raise ValueError(f"No GT examples found for category '{class_name}'")

    return gt_examples


# =============================================================================
# Get Negative Example for Class (for Stage 1b contrastive refinement)
# =============================================================================

def get_negative_example_for_class(coco_gt, train_dir, cat_id):
    """
    Get a random example from another class with red bounding boxes drawn.
    Used for Stage 1b contrastive refinement.

    Args:
        coco_gt: COCO ground truth object
        train_dir: Path to training images
        cat_id: Category ID of the negative class

    Returns:
        PIL image with red boxes drawn on the negative class objects
    """
    img_ids = coco_gt.getImgIds(catIds=[cat_id])
    if not img_ids:
        return None

    # Randomly choose 1 image
    chosen_img_id = random.choice(img_ids)

    ann_ids = coco_gt.getAnnIds(imgIds=[chosen_img_id], catIds=[cat_id])
    anns = coco_gt.loadAnns(ann_ids)

    img_info_list = coco_gt.loadImgs(chosen_img_id)
    if not img_info_list:
        return None

    img_info = img_info_list[0]
    image_path = os.path.join(train_dir, img_info["file_name"])
    if not os.path.isfile(image_path):
        return None

    # Get GT bboxes for this category
    gt_bboxes = [ann['bbox'] for ann in anns if ann['category_id'] == cat_id]
    if not gt_bboxes:
        return None

    # Load image and draw red boxes
    img = Image.open(image_path).convert("RGB")
    img_with_boxes = draw_colored_bboxes_on_image(img, "red", gt_bboxes)

    return img_with_boxes


# =============================================================================
# Calculate IoU (matches ipt/ipt_utils.py calculate_iou)
# =============================================================================

def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes in [x, y, w, h] COCO format.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert to xyxy
    box1_xyxy = [x1, y1, x1 + w1, y1 + h1]
    box2_xyxy = [x2, y2, x2 + w2, y2 + h2]

    # Calculate intersection
    xi1 = max(box1_xyxy[0], box2_xyxy[0])
    yi1 = max(box1_xyxy[1], box2_xyxy[1])
    xi2 = min(box1_xyxy[2], box2_xyxy[2])
    yi2 = min(box1_xyxy[3], box2_xyxy[3])

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    # Calculate union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


# =============================================================================
# Identify Worst Examples (matches ipt/run_bench_singleclass_IPT.py lines 720-968)
# =============================================================================

def identify_worst_examples(all_results, cat_id, class_name, coco_gt, train_dir,
                            prev_worst_examples_map, viz_dir=None, iteration=None):
    """
    Compare predictions to GT, identify worst FP/FN/TP examples.

    Args:
        all_results: List of per-image results dicts with keys:
            - 'img_id': Image ID
            - 'image_path': Path to image
            - 'gt_anns': List of GT annotations for this image
            - 'predictions': List of prediction dicts with 'bbox' and 'score'
        cat_id: Category ID being evaluated
        class_name: Class name
        coco_gt: COCO ground truth object
        train_dir: Path to training images
        prev_worst_examples_map: Previously used examples to exclude (for diversity)
        viz_dir: Optional directory to save visualization images (for debugging)
        iteration: Current iteration number (for naming saved images)

    Returns:
        tuple: (few_shot_examples dict, updated prev_worst_examples_map)

        few_shot_examples has keys:
        - 'best_match': PIL image with green box on best TP
        - 'worst_fp': PIL image with red box on worst FP
        - 'worst_fn': PIL image with blue box on worst FN
    """
    image_pred_performance = []

    for result in all_results:
        gt_bboxes = [ann['bbox'] for ann in result['gt_anns'] if ann['category_id'] == cat_id]
        other_cls_gt_bboxes = [ann['bbox'] for ann in result['gt_anns'] if ann['category_id'] != cat_id]
        pred_detections = result['predictions']

        # No predictions - check for false negatives
        if len(pred_detections) == 0:
            if len(gt_bboxes) > 0:
                for gt_bbox in gt_bboxes:
                    image_pred_performance.append({
                        "img_id": result['img_id'],
                        "gt_iou": 0.0,
                        "best_score": 0.0,
                        "other_cls_iou": 0.0,
                        "fp_error": 0.0,
                        "image_path": result['image_path'],
                        "gt_bbox": gt_bbox,
                        "pred_bbox": None,
                        "det_score": 0.0,
                    })
            continue

        for det in pred_detections:
            det_score = det.get('score', 1.0)
            pred_box = det['bbox']

            # Calculate IoU with other class GT boxes
            other_cls_iou = 0.0
            if other_cls_gt_bboxes:
                other_cls_ious = [calculate_iou(gt_box, pred_box) for gt_box in other_cls_gt_bboxes]
                other_cls_iou = max(other_cls_ious) if other_cls_ious else 0.0

            if len(gt_bboxes) == 0:
                # All predictions are false positives (no GT for this class)
                fp_error = det_score * max(0.2, other_cls_iou)
                image_pred_performance.append({
                    "img_id": result['img_id'],
                    "gt_iou": 0.0,
                    "best_score": -1.0,  # Flag for no GT boxes
                    "other_cls_iou": other_cls_iou,
                    "fp_error": fp_error,
                    "image_path": result['image_path'],
                    "gt_bbox": None,
                    "pred_bbox": pred_box,
                    "det_score": det_score,
                })
                continue

            # Normal case: there are both GT boxes and predictions
            gt_iou_list = [calculate_iou(gt_box, pred_box) for gt_box in gt_bboxes]
            best_gt_idx = gt_iou_list.index(max(gt_iou_list))
            gt_bbox = gt_bboxes[best_gt_idx]
            gt_iou = max(gt_iou_list)

            best_score = det_score * gt_iou

            if gt_iou > 0.0:
                # This prediction matches a GT box, cannot be FP
                fp_error = 0.0
            else:
                fp_error = det_score * max(0.2, other_cls_iou)

            image_pred_performance.append({
                "img_id": result['img_id'],
                "gt_iou": gt_iou,
                "best_score": best_score,
                "other_cls_iou": other_cls_iou,
                "fp_error": fp_error,
                "image_path": result['image_path'],
                "gt_bbox": gt_bbox,
                "pred_bbox": pred_box,
                "det_score": det_score,
            })

    # Select best match, worst FPs, worst FNs
    # Use top 3 FP/FN for more diverse refinement (Mitigation 5)
    num_examples = 3  # Number of FP/FN examples to use
    top_n = 10  # Candidate pool size for diversity

    # Best matches: highest best_score (still just 1 for reference)
    best_match_candidates = sorted(image_pred_performance, key=lambda x: x['best_score'], reverse=True)[:top_n]
    best_match_candidates = [c for c in best_match_candidates if c['best_score'] > 0.0]

    # Exclude previous best match image if multiple candidates exist
    if len(best_match_candidates) > 1 and 'best_match' in prev_worst_examples_map and prev_worst_examples_map['best_match']:
        prev_img_id = prev_worst_examples_map['best_match'].get('img_id')
        filtered = [c for c in best_match_candidates if c['img_id'] != prev_img_id]
        if filtered:
            best_match_candidates = filtered

    # Worst FPs: highest fp_error - get top N instead of just 1
    worst_fp_candidates = sorted(image_pred_performance, key=lambda x: x['fp_error'], reverse=True)[:top_n]
    worst_fp_candidates = [c for c in worst_fp_candidates if c['fp_error'] > 0.0]

    # Worst FNs: highest fn_error (1.0 - best_score for unmatched GT)
    worst_fn_candidates_dict = {}
    for c in image_pred_performance:
        if c['best_score'] == -1.0:  # No GT boxes, skip
            continue

        fn_error = 1.0 - c['best_score']
        c['fn_error'] = fn_error

        key = f"{c['img_id']}_{c['gt_bbox']}"
        if key not in worst_fn_candidates_dict:
            worst_fn_candidates_dict[key] = c
        else:
            # Keep the one with higher best_score
            if c['best_score'] > worst_fn_candidates_dict[key]['best_score']:
                worst_fn_candidates_dict[key] = c

    worst_fn_candidates = sorted(worst_fn_candidates_dict.values(), key=lambda x: x['fn_error'], reverse=True)[:top_n]
    worst_fn_candidates = [c for c in worst_fn_candidates if c['fn_error'] > 0.0]

    # Select examples: 1 best match, top N FPs, top N FNs
    best_match_example = random.choice(best_match_candidates) if best_match_candidates else None
    # Take top num_examples FPs (deterministic, not random) for diversity across error types
    worst_fp_examples = worst_fp_candidates[:num_examples] if worst_fp_candidates else []
    worst_fn_examples = worst_fn_candidates[:num_examples] if worst_fn_candidates else []

    # Update prev_worst_examples_map (track first example of each for logging)
    new_prev_worst_examples_map = {
        'best_match': best_match_example,
        'worst_fp': worst_fp_examples[0] if worst_fp_examples else None,
        'worst_fn': worst_fn_examples[0] if worst_fn_examples else None
    }

    # Build few_shot_examples with images
    # Now 'worst_fp' and 'worst_fn' are LISTS of images
    few_shot_examples = {}

    if best_match_example:
        img = Image.open(best_match_example['image_path']).convert("RGB")
        img_with_boxes = draw_colored_bboxes_on_image(img, "green", [best_match_example['gt_bbox']])
        few_shot_examples['best_match'] = img_with_boxes

        # Save visualization if viz_dir provided
        if viz_dir:
            iter_str = f"iter{iteration}_" if iteration is not None else ""
            viz_path = os.path.join(viz_dir, f"{class_name}_{iter_str}best_match_imgId{best_match_example['img_id']}_score{best_match_example['best_score']:.2f}.png")
            img_with_boxes.save(viz_path)

    # Build list of FP images
    if worst_fp_examples:
        fp_images = []
        for idx, fp_example in enumerate(worst_fp_examples):
            if fp_example and fp_example.get('pred_bbox'):
                img = Image.open(fp_example['image_path']).convert("RGB")
                img_with_boxes = draw_colored_bboxes_on_image(img, "red", [fp_example['pred_bbox']])
                fp_images.append(img_with_boxes)

                # Save visualization if viz_dir provided
                if viz_dir:
                    iter_str = f"iter{iteration}_" if iteration is not None else ""
                    viz_path = os.path.join(viz_dir, f"{class_name}_{iter_str}worst_fp{idx}_imgId{fp_example['img_id']}_error{fp_example['fp_error']:.2f}.png")
                    img_with_boxes.save(viz_path)

        if fp_images:
            few_shot_examples['worst_fp'] = fp_images  # Now a list!

    # Build list of FN images
    if worst_fn_examples:
        fn_images = []
        for idx, fn_example in enumerate(worst_fn_examples):
            if fn_example and fn_example.get('gt_bbox'):
                img = Image.open(fn_example['image_path']).convert("RGB")
                img_with_boxes = draw_colored_bboxes_on_image(img, "blue", [fn_example['gt_bbox']])
                fn_images.append(img_with_boxes)

                # Save visualization if viz_dir provided
                if viz_dir:
                    iter_str = f"iter{iteration}_" if iteration is not None else ""
                    viz_path = os.path.join(viz_dir, f"{class_name}_{iter_str}worst_fn{idx}_imgId{fn_example['img_id']}_error{fn_example['fn_error']:.2f}.png")
                    img_with_boxes.save(viz_path)

        if fn_images:
            few_shot_examples['worst_fn'] = fn_images  # Now a list!

    return few_shot_examples, new_prev_worst_examples_map


# =============================================================================
# mAP Computation (matches ipt/run_bench_singleclass_IPT.py lines 180-237)
# =============================================================================

def compute_map_for_class(predictions, coco_gt, cat_id):
    """
    Compute mAP for a single class using pycocotools COCOeval.

    Args:
        predictions: List of COCO-format detection dicts with keys:
            - image_id, category_id, bbox, score
        coco_gt: COCO ground truth object
        cat_id: Category ID to evaluate

    Returns:
        float: mAP@0.50:0.95 for this class
    """
    if not predictions:
        return 0.0

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.catIds = [cat_id]  # Single class evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats[0]  # mAP@0.50:0.95


# =============================================================================
# Checkpoint/Resume Support (matches ipt/run_bench_singleclass_IPT.py checkpoint logic)
# =============================================================================

def save_ipt_checkpoint(checkpoint_dir, dataset_name, class_name, iteration,
                        current_def, best_def, best_map, prev_mAP,
                        prev_instructions, instruction_refinements, prev_worst_examples_map=None):
    """Save IPT state for resume capability."""
    # Convert prev_worst_examples_map to serializable format (remove PIL images, keep metadata)
    serializable_prev_worst = {}
    if prev_worst_examples_map:
        for key, val in prev_worst_examples_map.items():
            if val is not None and isinstance(val, dict):
                # Only save the img_id for diversity tracking, not the full dict
                serializable_prev_worst[key] = {"img_id": val.get("img_id")}
            else:
                serializable_prev_worst[key] = None

    state = {
        "last_completed_iteration": iteration,
        "current_instructions": current_def,
        "best_instructions": best_def,
        "best_mAP": best_map,
        "prev_mAP": prev_mAP,
        "prev_instructions": prev_instructions,
        "instruction_refinements": instruction_refinements,
        "prev_worst_examples_map": serializable_prev_worst,
    }

    state_path = os.path.join(checkpoint_dir, f"ipt_state_{dataset_name}_{class_name}.json")
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def load_ipt_checkpoint(checkpoint_dir, dataset_name, class_name):
    """Load IPT state if exists, return None otherwise."""
    state_path = os.path.join(checkpoint_dir, f"ipt_state_{dataset_name}_{class_name}.json")

    if not os.path.exists(state_path):
        return None

    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    return state
