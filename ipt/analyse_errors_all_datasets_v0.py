"""
Rescores all 20 datasets using saved live_results, computes new_ranking and combined
detection scores, evaluates with COCO metrics, and consolidates results.

Usage:
    python rescore_all_datasets.py \
        --results_root results/results_data3/final_consolidated_results/rf-20-vl-benchmark/results/eccv26/rf100vl_IPT/Qwen2.5-VL-7B-Instruct/rf20_IPT_singleclass_rankScore \
        --dataset_root ./datasets/rf100-vl-fsod \
        --data_instr_path results/rf100vl_IPT/Qwen2.5-VL-7B-Instruct/rf20_IPT_singleclass_vqaScore_withNMS/iterative_prompt_refinement/all_refined_class_instructions
"""

import os
import json
import glob
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from rf100vl.util import get_basename, get_category

#Library to analyse detection errors in more detail (beyond COCO metrics)
# Source: https://github.com/dbolya/tide 
from tidecv import TIDE, datasets

# ---------------------------------------------------------------------------
# NMS helper (replicated from ipt_utils to avoid the dependency)
# ---------------------------------------------------------------------------

def compute_iou(box1, box2):
    """Compute IoU between two [x, y, w, h] boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def apply_nms(dets, iou_threshold=0.5, score_key="rank_score"):
    """Simple greedy NMS on a list of detection dicts."""
    if not dets:
        return dets
    dets = sorted(dets, key=lambda d: d.get(score_key, 0), reverse=True)
    keep = []
    suppressed = [False] * len(dets)
    for i in range(len(dets)):
        if suppressed[i]:
            continue
        keep.append(dets[i])
        for j in range(i + 1, len(dets)):
            if not suppressed[j]:
                if compute_iou(dets[i]["bbox"], dets[j]["bbox"]) >= iou_threshold:
                    suppressed[j] = True
    return keep


# ---------------------------------------------------------------------------
# Dataset list
# ---------------------------------------------------------------------------

DATASETS = [
    "aerial-airport",
    "dentalai",
    "flir-camera-objects",
    "gwhd2021",
    "recode-waste",
    "wildfire-smoke",
    "x-ray-id",
    "soda-bottles",
    "wb-prova",
    "actions",
    "aquarium-combined",
    "the-dreidel-project",
    "orionproducts",
    "trail-camera",
    "new-defects-in-wood",
    "lacrosse-object-detection",
    "defect-detection",
    "all-elements",
    "water-meter",
    "paper-parts",
]

METRIC_KEYS = [
    "AP_50_95", "AP_50", "AP_75", "AP_S", "AP_M", "AP_L",
    "AR_1", "AR_10", "AR_100", "AR_S", "AR_M", "AR_L",
]

# ---------------------------------------------------------------------------
# Core rescoring for one dataset
# ---------------------------------------------------------------------------

def rescore_dataset(tide, dataset_name: str, dataset_root: str, results_root: str,
                    data_instr_path: str, score_type: str = "rank",
                    nms_iou_threshold: float = 0.1, output_dir: str = None):
    """
    Loads saved live_results for *dataset_name*, computes new_ranking and
    combined scores, evaluates with COCO, and returns a dict of stats arrays.
    """
    # --- Locate ground truth annotations ---
    # Try exact name first, then glob for a fuzzy match
    dataset_path = os.path.join(dataset_root, dataset_name)
    if not os.path.isdir(dataset_path):
        candidates = glob.glob(os.path.join(dataset_root, f"{dataset_name}*"))
        if not candidates:
            print(f"[SKIP] Dataset directory not found for '{dataset_name}'")
            return None
        dataset_path = candidates[0]
        print(f"[INFO] Using dataset path: {dataset_path}")

    ann_path = os.path.join(dataset_path, "test", "_annotations.coco.json")
    if not os.path.isfile(ann_path):
        print(f"[SKIP] No test annotations at {ann_path}")
        return None

    coco_gt = COCO(ann_path)
    ds_cat_ids = coco_gt.getCatIds()
    cat_name2id = {coco_gt.cats[cid]["name"]: cid for cid in ds_cat_ids}

    # --- Locate predictions JSON ---
    # Pattern: <results_root>/final_instruction_eval/predictions/<score_type>/<dataset_name>_predictions.json
    if score_type == "model":
        predictions_path = os.path.join(
            results_root, "final_instruction_eval", "predictions",
            "rank", f"predictions_{dataset_name}_{score_type}.json"
        )
    else:
        predictions_path = os.path.join(
            results_root, "final_instruction_eval", "predictions",
            score_type, f"predictions_{dataset_name}_{score_type}.json"
        )

    if not os.path.isfile(predictions_path):
        # Fallback: search recursively
        found = glob.glob(
            os.path.join(results_root, "**", f"predictions_{dataset_name}_{score_type}.json"),
            recursive=True,
        )
        if not found:
            print(f"[SKIP] predictions not found for '{dataset_name}'")
            return None
        predictions_path = found[0]
        print(f"[INFO] Found predictions at: {predictions_path}")

    with open(predictions_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    detections_by_type = {}
    detections_by_type[score_type] = predictions


    # --- COCO evaluation ---
    all_stats = {}
    for eval_type, detections in detections_by_type.items():
        if not detections:
            print(f"  [WARN] No detections for eval_type='{eval_type}'")
            all_stats[eval_type] = [0.0] * 12
            continue
        # Filter out invalid category_ids
        valid = [d for d in detections if d["category_id"] != -1]
        if not valid:
            print(f"  [WARN] All detections have unknown category for '{eval_type}'")
            all_stats[eval_type] = [0.0] * 12
            continue
        try:
            coco_dt = coco_gt.loadRes(valid)
            coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            all_stats[eval_type] = coco_eval.stats.tolist()
        except Exception as e:
            print(f"  [ERROR] COCO eval failed for '{eval_type}': {e}")
            all_stats[eval_type] = [0.0] * 12

        try:
            # TIDE analysis (optional, can be time-consuming)
            tide_ds = datasets.COCO(ann_path)
            # tide_dt = datasets.COCOResult(valid)
            tide_dt = datasets.COCOResult(predictions_path)
            # tide = TIDE(tide_ds, tide_dt, iou_thresholds=[0.5], max_dets=100)

            # tide = TIDE()
            # tide.evaluate(tide_ds, tide_dt, mode=TIDE.BOX) #Use TIDE.BOX for bounding box evaluation, TIDE.MASK for segmentation masks
            
            tide.evaluate(tide_ds, tide_dt, mode=TIDE.BOX, name = dataset_name, use_for_errors=True) #Use TIDE.BOX for bounding box evaluation, TIDE.MASK for segmentation masks
            # tide.evaluate_range(tide_ds, tide_dt, mode=TIDE.BOX, name = dataset_name) #Use TIDE.BOX for bounding box evaluation, TIDE.MASK for segmentation masks
            
            # You can extract more detailed error types from tide.stats if desired
            # tide.summarize()
            # tide.plot(out_dir=os.path.join(output_dir, f"tide_analysis_{dataset_name}"))

        except Exception as e:
            print(f"  [ERROR] TIDE analysis failed for '{eval_type}': {e}")

    return all_stats


# ---------------------------------------------------------------------------
# Consolidation helpers (adapted from consolidate_results.py)
# ---------------------------------------------------------------------------

def consolidate(all_dataset_stats: dict[str, dict], output_dir: str):
    """
    all_dataset_stats: { dataset_name: { eval_type: [12 floats] } }
    Saves per-eval-type CSVs and a summary text file.
    """
    os.makedirs(output_dir, exist_ok=True)

    eval_types = set()
    for stats in all_dataset_stats.values():
        eval_types.update(stats.keys())

    summary_lines = []

    for eval_type in sorted(eval_types):
        records = []
        for ds_name, stats in all_dataset_stats.items():
            if eval_type not in stats:
                continue
            row = {"dataset": ds_name}
            for i, key in enumerate(METRIC_KEYS):
                row[key] = stats[eval_type][i] if i < len(stats[eval_type]) else 0.0
            records.append(row)

        if not records:
            continue

        df = pd.DataFrame(records)

        # Average row
        avg = df.drop(columns=["dataset"]).mean()
        avg_row = {"dataset": "AVERAGE", **avg.to_dict()}
        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

        csv_path = os.path.join(output_dir, f"results_{eval_type}.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n[{eval_type}] Saved: {csv_path}")

        # Category grouping
        cat_groups: dict[str, list] = defaultdict(list)
        for row in records:
            cat = get_category(row["dataset"])
            cat_groups[cat].append(row["AP_50_95"])

        mean_ap = avg["AP_50_95"]
        block = [
            f"\n=== {eval_type} ===",
            f"  Overall mAP (AP50-95): {mean_ap*100:.2f}",
        ]
        for cat, vals in sorted(cat_groups.items()):
            block.append(f"  {cat}: {np.mean(vals)*100:.2f}  (n={len(vals)})")

        # Per-dataset detail
        block.append("  --- Per dataset ---")
        for row in sorted(records, key=lambda r: r["dataset"]):
            block.append(f"    {row['dataset']}: {row['AP_50_95']*100:.2f}")

        summary_lines.extend(block)
        print("\n".join(block))

    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"\nSummary saved to: {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Rescore all 20 datasets and consolidate results.")
    parser.add_argument(
        "--results_root",
        type=str,
        # default="results/results_data3/final_consolidated_results/rf-20-vl-benchmark/"
        #         "results/eccv26/rf100vl_IPT/Qwen2.5-VL-7B-Instruct/rf20_IPT_singleclass_rankScore",
        # default="results/results_data3/final_consolidated_results/rf-20-vl-benchmark/"
        #         "results/eccv26/rf100vl_IPT/Qwen2.5-VL-72B-Instruct/rf20_IPT_singleclass_rankScore",
        # default="results/results_data3/final_consolidated_results/rf-20-vl-benchmark/"
        #         "results/eccv26/rf100vl_IPT/Qwen3-VL-8B-Instruct/rf20_IPT_singleclass_rankScore",
        default="results/results_data3/final_consolidated_results/rf-20-vl-benchmark/"
                "results/eccv26/rf100vl_IPT/Qwen3-VL-30B-A3B-Instruct/rf20_IPT_singleclass_rankScore",
        help="Root directory of the experiment run.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="./datasets/rf100-vl-fsod",
        help="Root directory containing all dataset folders.",
    )
    parser.add_argument(
        "--data_instr_path",
        type=str,
        default="results/rf100vl_IPT/Qwen2.5-VL-7B-Instruct/"
                "rf20_IPT_singleclass_vqaScore_withNMS/iterative_prompt_refinement/"
                "all_refined_class_instructions",
        help="Path prefix for per-dataset instruction JSON files.",
    )
    parser.add_argument(
        "--score_type",
        type=str,
        # default="rank",
        default="model",
        help="Sub-folder name under live_results/ (default: 'rank').",
    )
    parser.add_argument(
        "--nms_iou_threshold",
        type=float,
        default=0.1,
        help="IoU threshold for intra-class NMS before re-ranking.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to save consolidated CSVs. Defaults to <results_root>/consolidated_rescore/",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.results_root, "analyse_detection_errors")


    tide = TIDE()

    all_dataset_stats: dict[str, dict] = {}

    for dataset_name in DATASETS:
        print(f"\n{'='*60}")
        print(f"Processing: {dataset_name}")
        stats = rescore_dataset(
            tide=tide,
            dataset_name=dataset_name,
            dataset_root=args.dataset_root,
            results_root=args.results_root,
            data_instr_path=args.data_instr_path,
            score_type=args.score_type,
            nms_iou_threshold=args.nms_iou_threshold,
            output_dir=output_dir,
        )
        if stats is not None:
            all_dataset_stats[dataset_name] = stats
            for eval_type, s in stats.items():
                print(f"  [{eval_type}] AP50-95={s[0]*100:.2f}  AP50={s[1]*100:.2f}  AR@1={s[6]*100:.2f}")
        else:
            print(f"  [SKIP] No results for {dataset_name}")

    print(f"\n{'='*60}")
    print(f"Processed {len(all_dataset_stats)} / {len(DATASETS)} datasets.")
    consolidate(all_dataset_stats, output_dir)

    #Tide analysis

    tide.summarize()
    tide.plot(out_dir=os.path.join(output_dir, f"tide_analysis_{dataset_name}"))

    tide_erros = tide.get_all_errors()



if __name__ == "__main__":
    main()