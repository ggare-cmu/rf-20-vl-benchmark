"""
Batch rescore Gemini predictions using Qwen VLM across all datasets in a GCS experiment.

For each dataset found under --gcs_experiment/results/<dataset_name>/:
  - Downloads gemini_detection_results.json
  - Resolves local images from --datasets_root/<dataset_name>/test/
  - Resolves IPT instructions from --instructions_root/<dataset_name>/all_refined_class_instructions_<dataset_name>.json
    (falls back to data_instr/default/README.dataset_<dataset_name>.json if not found)
  - Runs Step 2 (1-5 rating) and Step 3 (VQA rescore on Step 2 results)
  - Uploads results to GCS as sibling folders (_step2_rating, _step3_vqa)

Usage:
  python rescore_gemini_preds_batch.py \
    --model_name Qwen3-VL-30B-A3B-Instruct \
    --gcs_experiment gs://rf-detr-rf100-vl/gemini_final_experiments/gemini3_ipt_fewshot \
    --datasets_root rf20-vl-fsod \
    --instructions_root gemini3_new_method \
    --max_model_len 16384
"""

import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import sys
import json
import gc
import argparse
import subprocess
import tempfile
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO

# Reuse everything from ipt_utils and single-dataset script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ipt'))
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from ipt_utils import (
    load_qwen_model,
    model_generate_with_scores,
    create_img_with_bbox,
    get_masked_image_vqa_scores_with_instructions,
)
from rescore_gemini_preds import (
    load_qwen_model_with_max_len,
    get_rating_logit_scores,
    save_coco_predictions,
    gsutil_run,
)


def discover_datasets(gcs_experiment):
    """List all dataset folders under gcs_experiment/results/."""
    gcs_results = f"{gcs_experiment.rstrip('/')}/results/"
    result = subprocess.run(
        ["gsutil", "ls", gcs_results],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  [ERR] Could not list {gcs_results}: {result.stderr.strip()}")
        return []

    # Each line is like gs://bucket/.../results/actions/
    datasets = []
    for line in result.stdout.strip().split("\n"):
        line = line.strip().rstrip("/")
        if not line:
            continue
        dataset_name = line.split("/")[-1]
        if dataset_name and dataset_name != "results":
            datasets.append(dataset_name)

    return datasets


def resolve_instructions(instructions_root, dataset_name):
    """
    Find the instructions JSON for a dataset. Tries in order:
      1. instructions_root/<dataset_name>/all_refined_class_instructions_<dataset_name>.json
      2. instructions_root/<dataset_name>/all_refined_class_instructions.json
      3. data_instr/default/README.dataset_<dataset_name>.json
    Returns path if found, None otherwise.
    """
    if instructions_root:
        candidate1 = os.path.join(
            instructions_root, dataset_name,
            f"all_refined_class_instructions_{dataset_name}.json"
        )
        if os.path.isfile(candidate1):
            return candidate1

        candidate2 = os.path.join(
            instructions_root, dataset_name,
            "all_refined_class_instructions.json"
        )
        if os.path.isfile(candidate2):
            return candidate2

    candidate3 = os.path.join(
        "data_instr", "default", f"README.dataset_{dataset_name}.json"
    )
    if os.path.isfile(candidate3):
        return candidate3

    return None


def rescore_dataset(qwen_model, qwen_processor, gcs_experiment, dataset_name,
                    datasets_root, instructions_json_path):
    """Rescore a single dataset: download preds, run Step 2 + Step 3, upload results."""

    gcs_input = f"{gcs_experiment.rstrip('/')}/results/{dataset_name}"
    images_dir = os.path.join(datasets_root, dataset_name, "test")
    gt_json = os.path.join(datasets_root, dataset_name, "test", "_annotations.coco.json")

    # Parse GCS path for upload destinations
    parts = gcs_input.replace("gs://", "").split("/")
    bucket = f"gs://{parts[0]}"
    try:
        results_idx = parts.index("results")
    except ValueError:
        print(f"  [ERR] No 'results' in path: {gcs_input}")
        return False

    run_name = "/".join(parts[1:results_idx])

    print(f"\n{'='*60}")
    print(f"  Dataset:      {dataset_name}")
    print(f"  GCS input:    {gcs_input}")
    print(f"  Images:       {images_dir}")
    print(f"  GT:           {gt_json}")
    print(f"  Instructions: {instructions_json_path}")
    print(f"  Run name:     {run_name}")
    print(f"{'='*60}")

    # ---- Download predictions ----
    local_tmp = tempfile.mkdtemp(prefix=f"rescore_{dataset_name}_")
    local_pred_json = os.path.join(local_tmp, "gemini_detection_results.json")
    gcs_pred_json = f"{gcs_input}/gemini_detection_results.json"

    print(f"\nDownloading {gcs_pred_json}...")
    if not gsutil_run(["gsutil", "cp", gcs_pred_json, local_pred_json]):
        return False

    # ---- Load data ----
    print("Loading predictions...")
    with open(local_pred_json, 'r') as f:
        predictions = json.load(f)
    print(f"  {len(predictions)} predictions loaded")

    if len(predictions) == 0:
        print("  No predictions, skipping.")
        return True

    print("Loading ground truth...")
    coco_gt = COCO(gt_json)
    img_id_to_filename = {img['id']: img['file_name'] for img in coco_gt.dataset['images']}
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_gt.dataset['categories']}
    print(f"  Categories: {cat_id_to_name}")

    print("Loading dataset instructions...")
    with open(instructions_json_path, 'r') as f:
        dataset_instructions = json.load(f)
    print(f"  Instruction keys: {list(dataset_instructions.keys())}")

    # ---- Prepare per-detection images ----
    print("\nPreparing images with bounding boxes...")
    pil_images_with_bbox = []
    prompt_list = []
    image_cache = {}

    for pred in tqdm(predictions, desc="Preparing bbox images"):
        img_id = pred['image_id']
        filename = img_id_to_filename[img_id]
        bbox_xywh = pred['bbox']
        cat_name = cat_id_to_name[pred['category_id']]

        if filename not in image_cache:
            img_path = os.path.join(images_dir, filename)
            image_cache[filename] = Image.open(img_path).convert("RGB")

        img_with_bbox = create_img_with_bbox(image_cache[filename], bbox_xywh)
        pil_images_with_bbox.append(img_with_bbox)
        prompt_list.append(cat_name)

    print(f"  {len(pil_images_with_bbox)} detection images prepared.")

    # ==== Step 2: 1-5 Rating with logit extraction ====
    print("\n=== Step 2: Qwen 1-5 Rating (logit-based) ===")
    rating_scores_raw = get_rating_logit_scores(
        qwen_model, qwen_processor, dataset_instructions,
        prompt_list, pil_images_with_bbox
    )

    rating_scores_norm = np.array([
        (r - 1.0) / 4.0 if r != -1.0 else predictions[i]['score']
        for i, r in enumerate(rating_scores_raw)
    ])

    valid_mask = rating_scores_raw != -1.0
    if valid_mask.any():
        print(f"  Raw rating scores: min={rating_scores_raw[valid_mask].min():.3f}, "
              f"max={rating_scores_raw[valid_mask].max():.3f}, "
              f"mean={rating_scores_raw[valid_mask].mean():.3f}, "
              f"num_failed={np.sum(~valid_mask)}")

    step2_local = os.path.join(local_tmp, "step2", "gemini_detection_results.json")
    save_coco_predictions(predictions, rating_scores_norm, step2_local)

    step2_gcs = f"{bucket}/{run_name}_step2_rating/results/{dataset_name}/gemini_detection_results.json"
    print(f"\n  Uploading Step 2 -> {step2_gcs}")
    gsutil_run(["gsutil", "cp", step2_local, step2_gcs])

    # ==== Step 3: VQA Yes/No Rescore on Step 2 results ====
    print("\n=== Step 3: VQA Yes/No Rescore (on Step 2 results) ===")
    vqa_scores = get_masked_image_vqa_scores_with_instructions(
        qwen_model, qwen_processor, dataset_instructions,
        prompt_list, pil_images_with_bbox, batch_size=1
    )

    # VQA replaces score; on failure, keep Step 2 rating score
    final_scores = np.array([
        float(vqa_scores[i]) if vqa_scores[i] != -1.0
        else float(rating_scores_norm[i])
        for i in range(len(predictions))
    ])

    vqa_valid_mask = vqa_scores != -1.0
    if vqa_valid_mask.any():
        print(f"  VQA scores: min={vqa_scores[vqa_valid_mask].min():.3f}, "
              f"max={vqa_scores[vqa_valid_mask].max():.3f}, "
              f"mean={vqa_scores[vqa_valid_mask].mean():.3f}, "
              f"num_failed={np.sum(~vqa_valid_mask)}")

    step3_local = os.path.join(local_tmp, "step3", "gemini_detection_results.json")
    save_coco_predictions(predictions, final_scores, step3_local)

    step3_gcs = f"{bucket}/{run_name}_step3_vqa/results/{dataset_name}/gemini_detection_results.json"
    print(f"\n  Uploading Step 3 -> {step3_gcs}")
    gsutil_run(["gsutil", "cp", step3_local, step3_gcs])

    # Free memory
    del pil_images_with_bbox, image_cache, prompt_list
    gc.collect()

    print(f"\n  Done with {dataset_name}!")
    print(f"    Step 2 (rating): {step2_gcs}")
    print(f"    Step 3 (vqa):    {step3_gcs}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Batch rescore Gemini predictions with Qwen")
    parser.add_argument("--model_name", type=str, default="Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--gcs_experiment", type=str, required=True,
                        help="GCS experiment folder, e.g. gs://rf-detr-rf100-vl/gemini_final_experiments/gemini3_ipt_fewshot")
    parser.add_argument("--datasets_root", type=str, required=True,
                        help="Local root for datasets, e.g. rf20-vl-fsod")
    parser.add_argument("--instructions_root", type=str, default=None,
                        help="Local root for IPT instructions, e.g. gemini3_new_method")
    parser.add_argument("--max_model_len", type=int, default=16384,
                        help="Max sequence length for vLLM KV cache")
    args = parser.parse_args()

    # ---- Discover datasets ----
    print(f"Discovering datasets under {args.gcs_experiment}/results/...")
    datasets = discover_datasets(args.gcs_experiment)
    print(f"  Found {len(datasets)} datasets: {datasets}")

    if not datasets:
        print("  Nothing to process.")
        return

    # ---- Validate all datasets before loading model ----
    valid_datasets = []
    for dataset_name in datasets:
        images_dir = os.path.join(args.datasets_root, dataset_name, "test")
        gt_json = os.path.join(args.datasets_root, dataset_name, "test", "_annotations.coco.json")
        instructions = resolve_instructions(args.instructions_root, dataset_name)

        if not os.path.isdir(images_dir):
            print(f"  [SKIP] {dataset_name} — images dir not found: {images_dir}")
            continue
        if not os.path.isfile(gt_json):
            print(f"  [SKIP] {dataset_name} — GT json not found: {gt_json}")
            continue
        if instructions is None:
            print(f"  [SKIP] {dataset_name} — no instructions JSON found")
            continue

        valid_datasets.append((dataset_name, instructions))
        print(f"  [OK]   {dataset_name} — instructions: {instructions}")

    if not valid_datasets:
        print("\nNo valid datasets to process.")
        return

    print(f"\n{len(valid_datasets)} datasets ready to process.")

    # ---- Load Qwen model once ----
    print(f"\nLoading Qwen model: {args.model_name} (max_model_len={args.max_model_len})...")
    qwen_model, qwen_processor = load_qwen_model_with_max_len(args.model_name, max_model_len=args.max_model_len)
    print("  Model loaded.\n")

    # ---- Process each dataset ----
    succeeded = 0
    failed = 0

    for dataset_name, instructions_path in valid_datasets:
        try:
            ok = rescore_dataset(
                qwen_model, qwen_processor,
                args.gcs_experiment, dataset_name,
                args.datasets_root, instructions_path
            )
            if ok:
                succeeded += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n[ERR] Failed on {dataset_name}: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Batch complete: {succeeded} succeeded, {failed} failed, {len(datasets) - len(valid_datasets)} skipped")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
