"""
Rescore Gemini predictions using Qwen VLM in two steps:
  Step 1: 1-5 quality rating with logit-based confidence extraction (NEW)
  Step 2: VQA Yes/No rescore (REUSED identically from ipt_utils.py)

Each step saves results in dbl_eval.py-compatible layout:
  <output_dir>_step1_rating/results/actions/gemini_detection_results.json
  <output_dir>_step2_vqa/results/actions/gemini_detection_results.json

Usage:
  python rescore_gemini_preds.py \
    --model_name Qwen2.5-VL-7B-Instruct \
    --predictions_json /home/matveipopov/rf-20-vl-benchmark/gemini3_reg_preds_ranked/results/actions/gemini_detection_results.json \
    --images_dir /home/matveipopov/rf-20-vl-benchmark/rf20-vl-fsod/actions/test \
    --gt_json /home/matveipopov/rf-20-vl-benchmark/rf20-vl-fsod/actions/test/_annotations.coco.json \
    --instructions_json data_instr/default/README.dataset_actions.json \
    --dataset_name actions \
    --output_dir rescored_results
"""

import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import sys
import json
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO

# Reuse everything from ipt_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ipt'))
from ipt_utils import (
    load_qwen_model,
    model_generate_with_scores,
    create_img_with_bbox,
    get_masked_image_vqa_scores_with_instructions,
)


# ---- Step 1: NEW function ----
def get_rating_logit_scores(qwen_model, qwen_processor, dataset_instructions_json, prompt_list, pil_images):
    """
    Ask Qwen to rate each detection on a scale of 1-5, extract logprobs for
    digit tokens "1"-"5", and compute a probability-weighted expected score.

    Uses model_generate_with_scores from ipt_utils identically to VQA scoring,
    but with a rating prompt and digit-token logprob extraction instead of Yes/No.

    Returns: np.ndarray of shape (len(pil_images),) with scores in [1, 5] range,
             or -1.0 for failures.
    """

    # Reuse the same case-insensitive instruction lookup from ipt_utils
    def getDatasetInstructions(dataset_instructions_json, class_name):
        if class_name in dataset_instructions_json:
            return dataset_instructions_json[class_name]
        matched_key = next((key for key in dataset_instructions_json.keys() if key.lower() == class_name.lower()), None)
        if matched_key:
            return dataset_instructions_json[matched_key]
        raise ValueError(f"Class name '{class_name}' not found in dataset instructions JSON keys.")

    def getRatingPrompt(prompt, dataset_instructions_json):
        return f"""Given the '{prompt}' class defined as follows: {getDatasetInstructions(dataset_instructions_json, prompt)}

How well does the object inside the red bounding box match the class '{prompt}'? Rate the quality of this detection on a scale of 1 to 5, where:
1 = Not at all (wrong object or empty box)
2 = Poor match (partially visible or very unclear)
3 = Moderate match (somewhat matches but uncertain)
4 = Good match (clearly the right object)
5 = Excellent match (perfect detection)

Answer with a single number from 1 to 5."""

    digit_tokens = ["1", "2", "3", "4", "5"]
    all_scores = []

    for i in tqdm(range(len(pil_images)), desc="Step 1: Rating 1-5"):
        img = pil_images[i]
        prompt = prompt_list[i]

        # Same message format as VQA scoring in ipt_utils
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": getRatingPrompt(prompt, dataset_instructions_json)}
        ]}]

        # Reuse model_generate_with_scores identically (max_new_tokens=1, logprobs=5)
        outputs = model_generate_with_scores(messages, qwen_model, qwen_processor, max_new_tokens=1)

        # Same logprob access pattern as ipt_utils line 360
        token_logprobs = outputs[0].outputs[0].logprobs[0]

        # Extract logprobs for digit tokens 1-5
        digit_logprobs = {}
        for token_id, token_info in token_logprobs.items():
            decoded = token_info.decoded_token.strip()
            if decoded in digit_tokens:
                digit_logprobs[decoded] = token_info.logprob

        if not digit_logprobs:
            print(f"  [WARN] No digit tokens in top logprobs for detection {i}, assigning score -1.0")
            all_scores.append(-1.0)
            continue

        # Convert logprobs to probabilities (same pattern as ipt_utils exp())
        digits_found = sorted(digit_logprobs.keys())
        logprobs_tensor = torch.tensor([digit_logprobs[d] for d in digits_found])
        probs = torch.exp(logprobs_tensor)

        # Normalize over found digits
        probs = probs / (probs.sum() + 1e-18)

        # Expected value: weighted average of digit values
        digit_values = torch.tensor([float(d) for d in digits_found])
        expected_score = (probs * digit_values).sum().item()

        all_scores.append(expected_score)

    return np.array(all_scores)


def save_coco_predictions(predictions, scores, output_path):
    """Save predictions with updated scores in COCO format for dbl_eval.py."""
    coco_preds = []
    for i, pred in enumerate(predictions):
        coco_preds.append({
            "image_id": pred['image_id'],
            "category_id": pred['category_id'],
            "bbox": pred['bbox'],
            "score": float(scores[i])
        })
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco_preds, f, indent=2)
    print(f"  Saved {len(coco_preds)} predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Rescore Gemini predictions with Qwen (1-5 rating + VQA)")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--predictions_json", type=str, required=True,
                        help="Path to gemini_detection_results.json")
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Path to test images folder")
    parser.add_argument("--gt_json", type=str, required=True,
                        help="Path to _annotations.coco.json")
    parser.add_argument("--instructions_json", type=str, required=True,
                        help="Path to dataset instructions JSON")
    parser.add_argument("--dataset_name", type=str, default="actions",
                        help="Dataset name (used for output folder structure)")
    parser.add_argument("--output_dir", type=str, default="rescored_results",
                        help="Base output directory")
    args = parser.parse_args()

    # ---- Load data ----
    print("Loading predictions...")
    with open(args.predictions_json, 'r') as f:
        predictions = json.load(f)
    print(f"  {len(predictions)} predictions loaded")

    print("Loading ground truth for image_id -> filename mapping...")
    coco_gt = COCO(args.gt_json)
    img_id_to_filename = {img['id']: img['file_name'] for img in coco_gt.dataset['images']}
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_gt.dataset['categories']}
    print(f"  Categories: {cat_id_to_name}")

    print("Loading dataset instructions...")
    with open(args.instructions_json, 'r') as f:
        dataset_instructions = json.load(f)
    print(f"  Instruction keys: {list(dataset_instructions.keys())}")

    # ---- Load Qwen model (reuse load_qwen_model from ipt_utils) ----
    print(f"\nLoading Qwen model: {args.model_name}...")
    qwen_model, qwen_processor = load_qwen_model(args.model_name)
    print("  Model loaded.")

    # ---- Prepare per-detection images (reuse create_img_with_bbox from ipt_utils) ----
    print("\nPreparing images with bounding boxes...")
    pil_images_with_bbox = []
    prompt_list = []
    image_cache = {}

    for pred in tqdm(predictions, desc="Preparing bbox images"):
        img_id = pred['image_id']
        filename = img_id_to_filename[img_id]
        bbox_xywh = pred['bbox']  # already [x, y, w, h]
        cat_name = cat_id_to_name[pred['category_id']]

        if filename not in image_cache:
            img_path = os.path.join(args.images_dir, filename)
            image_cache[filename] = Image.open(img_path).convert("RGB")

        img_with_bbox = create_img_with_bbox(image_cache[filename], bbox_xywh)
        pil_images_with_bbox.append(img_with_bbox)
        prompt_list.append(cat_name)

    print(f"  {len(pil_images_with_bbox)} detection images prepared.")

    # ==== Step 1: 1-5 Rating with logit extraction (NEW) ====
    print("\n=== Step 1: Qwen 1-5 Rating (logit-based) ===")
    rating_scores_raw = get_rating_logit_scores(
        qwen_model, qwen_processor, dataset_instructions,
        prompt_list, pil_images_with_bbox
    )

    # Normalize 1-5 -> 0-1 for COCO eval, keep -1.0 failures as original score
    rating_scores_norm = np.array([
        (r - 1.0) / 4.0 if r != -1.0 else predictions[i]['score']
        for i, r in enumerate(rating_scores_raw)
    ])

    print(f"  Raw rating scores: min={rating_scores_raw[rating_scores_raw != -1.0].min():.3f}, "
          f"max={rating_scores_raw[rating_scores_raw != -1.0].max():.3f}, "
          f"mean={rating_scores_raw[rating_scores_raw != -1.0].mean():.3f}, "
          f"num_failed={np.sum(rating_scores_raw == -1.0)}")

    # Save Step 1 in dbl_eval.py layout: <output_dir>_step1_rating/results/<dataset>/gemini_detection_results.json
    step1_dir = f"{args.output_dir}_step1_rating"
    step1_path = os.path.join(step1_dir, "results", args.dataset_name, "gemini_detection_results.json")
    save_coco_predictions(predictions, rating_scores_norm, step1_path)

    # ==== Step 2: VQA Yes/No Rescore (REUSED identically from ipt_utils) ====
    print("\n=== Step 2: VQA Yes/No Rescore ===")
    vqa_scores = get_masked_image_vqa_scores_with_instructions(
        qwen_model, qwen_processor, dataset_instructions,
        prompt_list, pil_images_with_bbox, batch_size=1
    )

    # For failures, fall back to rating score, then original
    final_scores = np.array([
        float(vqa_scores[i]) if vqa_scores[i] != -1.0
        else float(rating_scores_norm[i])
        for i in range(len(predictions))
    ])

    print(f"  VQA scores: min={vqa_scores[vqa_scores != -1.0].min():.3f}, "
          f"max={vqa_scores[vqa_scores != -1.0].max():.3f}, "
          f"mean={vqa_scores[vqa_scores != -1.0].mean():.3f}, "
          f"num_failed={np.sum(vqa_scores == -1.0)}")

    # Save Step 2 in dbl_eval.py layout: <output_dir>_step2_vqa/results/<dataset>/gemini_detection_results.json
    step2_dir = f"{args.output_dir}_step2_vqa"
    step2_path = os.path.join(step2_dir, "results", args.dataset_name, "gemini_detection_results.json")
    save_coco_predictions(predictions, final_scores, step2_path)

    print(f"\nDone! To evaluate with dbl_eval.py:")
    print(f"  Step 1 only:  python /home/matveipopov/rf-20-vl-benchmark/dbl_eval.py --predictions_root {step1_dir} --model_name qwen_rating")
    print(f"  Step 1 + 2:   python /home/matveipopov/rf-20-vl-benchmark/dbl_eval.py --predictions_root {step2_dir} --model_name qwen_vqa")


if __name__ == "__main__":
    main()
