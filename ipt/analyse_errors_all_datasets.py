"""
Rescores all 20 datasets using saved live_results, computes new_ranking and combined
detection scores, evaluates with COCO metrics, and consolidates results.

Usage:
    python rescore_all_datasets.py \
        --results_root results/results_data3/final_consolidated_results/rf-20-vl-benchmark/results/eccv26/rf100vl_IPT/Qwen2.5-VL-7B-Instruct/rf20_IPT_singleclass_rankScore \
        --dataset_root ./datasets/rf100-vl-fsod \
        --data_instr_path results/rf100vl_IPT/Qwen2.5-VL-7B-Instruct/rf20_IPT_singleclass_vqaScore_withNMS/iterative_prompt_refinement/all_refined_class_instructions
"""

import math
import os
import json
import glob
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from rf100vl.util import get_basename, get_category

# Library to analyse detection errors in more detail (beyond COCO metrics)
# Source: https://github.com/dbolya/tide
from tidecv import TIDE, datasets
from tidecv.plotting import Plotter

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
    Loads saved predictions for *dataset_name*, evaluates with COCO metrics
    and TIDE error analysis, and returns a dict of COCO stats arrays.
    """
    # --- Locate ground truth annotations ---
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
    cat_name2id = {coco_gt.cats[cid]["name"]: cid for cid in coco_gt.getCatIds()}

    # --- Locate predictions JSON ---
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

    # --- COCO evaluation ---
    all_stats = {}
    valid = [d for d in predictions if d.get("category_id", -1) != -1]

    if not valid:
        print(f"  [WARN] All detections have unknown category for '{score_type}'")
        all_stats[score_type] = [0.0] * 12
    else:
        try:
            coco_dt   = coco_gt.loadRes(valid)
            coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            all_stats[score_type] = coco_eval.stats.tolist()
        except Exception as e:
            print(f"  [ERROR] COCO eval failed for '{score_type}': {e}")
            all_stats[score_type] = [0.0] * 12

    # --- TIDE evaluation ---
    try:
        tide_ds = datasets.COCO(ann_path)
        tide_dt = datasets.COCOResult(predictions_path)
        tide.evaluate(tide_ds, tide_dt, mode=TIDE.BOX, name=dataset_name, use_for_errors=True)
    except Exception as e:
        print(f"  [ERROR] TIDE analysis failed for '{dataset_name}': {e}")

    return all_stats


# ---------------------------------------------------------------------------
# TIDE error consolidation
# ---------------------------------------------------------------------------

def consolidate_tide_errors(errors_per_dataset: dict) -> dict:
    """
    Average main and special TIDE errors across all datasets.

    Parameters
    ----------
    errors_per_dataset : dict
        Shape produced by tide.get_all_errors():
        {
          'main':    { dataset_name: { error_type: float } },
          'special': { dataset_name: { error_type: float } },
        }

    Returns
    -------
    dict with a single 'consolidated' key per section:
        {
          'main':    { 'consolidated': { error_type: mean_float } },
          'special': { 'consolidated': { error_type: mean_float } },
        }
    """
    consolidated = {}
    for err_section in ("main", "special"):
        ds_errors = errors_per_dataset.get(err_section, {})
        if not ds_errors:
            consolidated[err_section] = {"consolidated": {}}
            continue

        accumulated: dict[str, list] = defaultdict(list)
        for ds_name, error_dict in ds_errors.items():
            for key, val in error_dict.items():
                accumulated[key].append(val)

        first_keys = list(next(iter(ds_errors.values())).keys())
        consolidated[err_section] = {
            "consolidated": {k: float(np.mean(accumulated[k])) for k in first_keys}
        }

    return consolidated


# ---------------------------------------------------------------------------
# TIDE consolidated plot
# ---------------------------------------------------------------------------

def plot_consolidated_tide_errors(
    errors_per_dataset: dict,
    out_dir: str,
    model_name: str = "consolidated",
    rec_type: str = "bbox",
) -> dict:
    """
    Produce two figures from per-dataset TIDE errors and return the
    consolidated (mean) error dict.

    Outputs
    -------
    <out_dir>/<model_name>_<rec_type>_consolidated_summary.png
        Pie + horizontal bars + FP/FN vertical bars.
        Reuses Plotter.make_summary_plot() for layout consistency.

    <out_dir>/<model_name>_<rec_type>_per_dataset_spread.png
        Mean ± std bars with individual dataset values as scatter points.
        Shows how consistent errors are across the 20 datasets.

    Parameters
    ----------
    errors_per_dataset : dict
        Raw per-dataset errors from tide.get_all_errors().
    out_dir : str
        Output directory (created if absent).
    model_name : str
        Label used in plot titles and filenames.
    rec_type : str
        'bbox' or 'segm' — used in filenames only.

    Returns
    -------
    consolidated : dict  — averaged error dict (single 'consolidated' key).
    """
    os.makedirs(out_dir, exist_ok=True)

    consolidated = consolidate_tide_errors(errors_per_dataset)
    n_datasets   = len(errors_per_dataset.get("main", {}))

    print(f"\n[TIDE] Consolidated errors across {n_datasets} datasets:")
    for k, v in consolidated["main"]["consolidated"].items():
        print(f"  {k:10s}: {v:.3f}")
    for k, v in consolidated["special"]["consolidated"].items():
        print(f"  {k:12s}: {v:.3f}")

    # ------------------------------------------------------------------ #
    # Figure 1: reuse Plotter.make_summary_plot()                        #
    # ------------------------------------------------------------------ #
    plotter = Plotter()

    dap_granularity = 5
    main_vals    = list(consolidated["main"]["consolidated"].values())
    special_vals = list(consolidated["special"]["consolidated"].values())
    max_main     = max(main_vals)    if main_vals    else plotter.MAX_MAIN_DELTA_AP
    max_special  = max(special_vals) if special_vals else plotter.MAX_SPECIAL_DELTA_AP
    if max_main > plotter.MAX_MAIN_DELTA_AP:
        plotter.MAX_MAIN_DELTA_AP    = math.ceil(max_main    / dap_granularity) * dap_granularity
    if max_special > plotter.MAX_SPECIAL_DELTA_AP:
        plotter.MAX_SPECIAL_DELTA_AP = math.ceil(max_special / dap_granularity) * dap_granularity

    remapped = {
        "main":    {model_name: consolidated["main"]["consolidated"]},
        "special": {model_name: consolidated["special"]["consolidated"]},
    }
    plotter.make_summary_plot(out_dir, remapped, model_name, rec_type, hbar_names=True)

    # Rename so it doesn't collide with per-dataset plots
    src = os.path.join(out_dir, f"{model_name}_{rec_type}_summary.png")
    dst = os.path.join(out_dir, f"{model_name}_{rec_type}_consolidated_summary.png")
    if os.path.isfile(src) and src != dst:
        os.replace(src, dst)
    print(f"[TIDE] Saved consolidated summary -> {dst}")

    # ------------------------------------------------------------------ #
    # Figure 2: per-dataset spread                                        #
    # ------------------------------------------------------------------ #
    _plot_per_dataset_spread(
        errors_per_dataset=errors_per_dataset,
        out_dir=out_dir,
        model_name=model_name,
        rec_type=rec_type,
        colors_main=plotter.colors_main,
        colors_special=plotter.colors_special,
        n_datasets=n_datasets,
    )

    return consolidated


def _plot_per_dataset_spread(
    errors_per_dataset: dict,
    out_dir: str,
    model_name: str,
    rec_type: str,
    colors_main: OrderedDict,
    colors_special: OrderedDict,
    n_datasets: int,
):
    """
    Horizontal bar chart (mean ± std) with individual dataset values
    overlaid as black scatter points for each TIDE error type.
    """
    main_ds    = errors_per_dataset.get("main",    {})
    special_ds = errors_per_dataset.get("special", {})

    main_rows = [
        {"Dataset": ds, "Error Type": err, "Delta mAP": val}
        for ds, err_dict in main_ds.items()
        for err, val in err_dict.items()
    ]
    special_rows = [
        {"Dataset": ds, "Error Type": err, "Delta mAP": val}
        for ds, err_dict in special_ds.items()
        for err, val in err_dict.items()
    ]
    empty_cols = ["Dataset", "Error Type", "Delta mAP"]
    main_df    = pd.DataFrame(main_rows)    if main_rows    else pd.DataFrame(columns=empty_cols)
    special_df = pd.DataFrame(special_rows) if special_rows else pd.DataFrame(columns=empty_cols)

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 6), dpi=150,
        gridspec_kw={"width_ratios": [3, 1]},
    )
    fig.suptitle(
        f"{model_name} — TIDE error breakdown across {n_datasets} datasets",
        fontsize=13, fontweight="bold", y=1.02,
    )

    # ---- Main errors ----
    ax_main = axes[0]
    if not main_df.empty:
        error_order = list(colors_main.keys())
        sns.barplot(
            data=main_df, x="Delta mAP", y="Error Type",
            order=error_order, palette=dict(colors_main),
            estimator=np.mean, errorbar="sd",
            ax=ax_main, capsize=0.3, errwidth=1.5,
        )
        sns.stripplot(
            data=main_df, x="Delta mAP", y="Error Type",
            order=error_order,
            color="black", alpha=0.4, size=4, jitter=False, ax=ax_main,
        )
        mean_vals = main_df.groupby("Error Type")["Delta mAP"].mean()
        for i, err_type in enumerate(error_order):
            if err_type in mean_vals:
                ax_main.text(
                    mean_vals[err_type] + 0.15, i,
                    f"{mean_vals[err_type]:.2f}",
                    va="center", fontsize=9, color="#333333",
                )
    ax_main.set_xlabel("Δ mAP", fontsize=11)
    ax_main.set_ylabel("")
    ax_main.set_title("Main error types (mean ± std)", fontsize=11)
    sns.despine(ax=ax_main, left=True, bottom=True)

    # ---- Special errors ----
    ax_spec = axes[1]
    if not special_df.empty:
        error_order_s = list(colors_special.keys())
        sns.barplot(
            data=special_df, x="Error Type", y="Delta mAP",
            order=error_order_s, palette=dict(colors_special),
            estimator=np.mean, errorbar="sd",
            ax=ax_spec, capsize=0.3, errwidth=1.5,
        )
        sns.stripplot(
            data=special_df, x="Error Type", y="Delta mAP",
            order=error_order_s,
            color="black", alpha=0.4, size=4, jitter=False, ax=ax_spec,
        )
        mean_vals_s = special_df.groupby("Error Type")["Delta mAP"].mean()
        for i, err_type in enumerate(error_order_s):
            if err_type in mean_vals_s:
                ax_spec.text(
                    i, mean_vals_s[err_type] + 0.3,
                    f"{mean_vals_s[err_type]:.2f}",
                    ha="center", fontsize=9, color="#333333",
                )
    ax_spec.set_xlabel("")
    ax_spec.set_ylabel("Δ mAP", fontsize=11)
    ax_spec.set_title("FP / FN (mean ± std)", fontsize=11)
    sns.despine(ax=ax_spec, left=True, bottom=True)

    plt.tight_layout()
    spread_path = os.path.join(out_dir, f"{model_name}_{rec_type}_per_dataset_spread.png")
    plt.savefig(spread_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[TIDE] Saved spread plot -> {spread_path}")


# ---------------------------------------------------------------------------
# Save / load TIDE errors
# ---------------------------------------------------------------------------

def save_tide_errors(errors: dict, path: str):
    """Persist tide.get_all_errors() to JSON for later re-use without re-running eval."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2)
    print(f"[TIDE] Errors saved -> {path}")


def load_tide_errors(path: str) -> dict:
    """Reload previously saved TIDE errors JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# COCO metric consolidation
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
        avg = df.drop(columns=["dataset"]).mean()
        avg_row = {"dataset": "AVERAGE", **avg.to_dict()}
        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

        csv_path = os.path.join(output_dir, f"results_{eval_type}.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n[{eval_type}] Saved: {csv_path}")

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
        default="model",
        help="Score type to evaluate: 'model', 'rank', etc.",
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
        help="Where to save outputs. Defaults to <results_root>/analyse_detection_errors/",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name for plot titles / filenames. Inferred from results_root if not set.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.results_root, "analyse_detection_errors", args.score_type)
    os.makedirs(output_dir, exist_ok=True)

    # Infer model name from path if not provided (e.g. 'Qwen3-VL-30B-A3B-Instruct')
    model_name = args.model_name or Path(args.results_root).parts[-2]

    # ------------------------------------------------------------------ #
    # 1. Per-dataset COCO eval + TIDE evaluation                          #
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # 2. COCO metric consolidation → CSVs + summary.txt                  #
    # ------------------------------------------------------------------ #
    consolidate(all_dataset_stats, output_dir)

    # ------------------------------------------------------------------ #
    # 3. TIDE: terminal summary + per-dataset plots                       #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print("TIDE summary across all datasets:")
    tide.summarize()

    tide_plots_dir = os.path.join(output_dir, "tide_plots_per_dataset")
    tide.plot(out_dir=tide_plots_dir)
    print(f"[TIDE] Per-dataset plots saved -> {tide_plots_dir}")

    # ------------------------------------------------------------------ #
    # 4. Consolidated TIDE error analysis + plots                         #
    # ------------------------------------------------------------------ #
    tide_errors = tide.get_all_errors()

    # Persist raw errors — allows re-plotting without re-running eval
    errors_json_path = os.path.join(output_dir, f"tide_errors_{args.score_type}.json")
    save_tide_errors(tide_errors, errors_json_path)

    consolidated_errors = plot_consolidated_tide_errors(
        errors_per_dataset=tide_errors,
        out_dir=os.path.join(output_dir, "tide_plots_consolidated"),
        model_name=model_name,
        rec_type="bbox",
    )

    # ------------------------------------------------------------------ #
    # 5. Print consolidated error table to terminal                       #
    # ------------------------------------------------------------------ #

    #Print and save to text file 
    # Save to text file
    table_path = os.path.join(output_dir, f"{model_name}_consolidated_tide_errors.txt")
    with open(table_path, "w") as f:
        f.write(f"{'='*60}\n")
        f.write(f"Consolidated TIDE errors for {model_name}:\n")
        all_errors = {
            **consolidated_errors["main"]["consolidated"],
            **consolidated_errors["special"]["consolidated"],
        }
        col_w = [
            max(len("Error Type"), max(len(k) for k in all_errors)),
            max(len("Mean Δ mAP"), 9),
        ]
        div = "  " + "---".join("-" * w for w in col_w)
        f.write(div.replace("-", "=") + "\n")
        f.write("  " + "   ".join([f"{'Error Type':>{col_w[0]}}", f"{'Mean Δ mAP':>{col_w[1]}}"]) + "\n")
        f.write(div + "\n")
        for k, v in all_errors.items():
            f.write("  " + "   ".join([f"{k:>{col_w[0]}}", f"{v:>{col_w[1]}.3f}"]) + "\n")
        f.write(div.replace("-", "=") + "\n") 
        
    print(f"\n{'='*60}")
    print(f"Consolidated TIDE errors for {model_name}:")
    all_errors = {
        **consolidated_errors["main"]["consolidated"],
        **consolidated_errors["special"]["consolidated"],
    }
    col_w = [
        max(len("Error Type"), max(len(k) for k in all_errors)),
        max(len("Mean Δ mAP"), 9),
    ]
    div = "  " + "---".join("-" * w for w in col_w)
    print(div.replace("-", "="))
    print("  " + "   ".join([f"{'Error Type':>{col_w[0]}}", f"{'Mean Δ mAP':>{col_w[1]}}"]))
    print(div)
    for k, v in all_errors.items():
        print("  " + "   ".join([f"{k:>{col_w[0]}}", f"{v:>{col_w[1]}.3f}"]))
    print(div.replace("-", "="))




if __name__ == "__main__":
    main()