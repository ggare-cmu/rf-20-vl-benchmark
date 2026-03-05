"""
Rescores all 20 datasets using saved predictions, evaluates with COCO metrics,
and computes + plots a per-class confusion matrix for each dataset.

Confusion matrix definition used here
--------------------------------------
For each image we first filter predicted boxes by IoU ≥ iou_threshold (default
0.5) against any ground-truth box.  Only predictions that survive this spatial
filter are considered "localised" and contribute to TP/FP counts.

  TP  predicted class C, matched GT box has class C      (correct class + localised)
  FP  predicted class C, matched GT box has class C'≠C   (wrong class  + localised)
       — these are the off-diagonal cells                —
  FN  GT box with class C has no localised prediction    (missed detection)

Unmatched predictions (IoU < threshold with every GT box) are treated as
background false-positives and are tallied in a separate "background" row/column
so the matrix remains square.

Usage
-----
  python rescore_all_datasets.py \
      --results_root results/.../rf20_IPT_singleclass_rankScore \
      --dataset_root ./datasets/rf100-vl-fsod \
      --score_type model
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
from itertools import product

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from rf100vl.util import get_basename, get_category

# Library to analyse detection errors in more detail (beyond COCO metrics)
# Source: https://github.com/dbolya/tide
from tidecv import TIDE, datasets
from tidecv.plotting import Plotter

# ---------------------------------------------------------------------------
# NMS helper
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
# Confusion matrix computation
# ---------------------------------------------------------------------------

def compute_confusion_matrix(
    coco_gt: COCO,
    predictions: list,
    iou_threshold: float = 0.5,
) -> tuple[np.ndarray, list]:
    """
    Build a confusion matrix from COCO ground-truth and a predictions list.

    Algorithm
    ---------
    For every image:
      1. Collect GT annotations and predicted boxes.
      2. For each predicted box (sorted by score desc), find the highest-IoU
         GT box that has not been claimed yet.
         • If best IoU ≥ iou_threshold  → the prediction is "localised":
               pred_class == gt_class   → TP  (diagonal)
               pred_class != gt_class   → FP  (off-diagonal, confusion)
           The matched GT box is marked as claimed.
         • If best IoU < iou_threshold  → the prediction is unmatched
               → tallied as background FP  (column "background", row pred_class)
      3. Unclaimed GT boxes → FN for their class
               → tallied as background FN  (row "background", column gt_class)

    Parameters
    ----------
    coco_gt       : COCO ground-truth object
    predictions   : list of dicts with keys image_id, category_id, bbox, score
    iou_threshold : minimum IoU for a prediction to be considered localised

    Returns
    -------
    cm      : np.ndarray  shape (n_classes+1, n_classes+1)
              rows = predicted class (last row = background/unmatched GT)
              cols = GT class        (last col = background/unmatched pred)
              cm[i, j]  = number of times class i was predicted for a box
                          whose best-matching GT had class j
              cm[-1, j] = GT class j boxes with no localised prediction (FN)
              cm[i, -1] = predictions of class i with no GT match       (FP-bkg)
    labels  : list of class names, last entry is "background"
    """
    cat_ids   = sorted(coco_gt.getCatIds())
    cat_names = [coco_gt.cats[cid]["name"] for cid in cat_ids]
    catid2idx = {cid: i for i, cid in enumerate(cat_ids)}
    n_cls     = len(cat_ids)
    BKG       = n_cls   # index for the "background / unmatched" pseudo-class

    # n_cls+1 × n_cls+1:  cm[pred_idx, gt_idx]
    cm = np.zeros((n_cls + 1, n_cls + 1), dtype=np.int64)

    # Group predictions and GT by image
    img_ids = coco_gt.getImgIds()

    # Index predictions by image_id
    pred_by_img: dict[int, list] = defaultdict(list)
    for p in predictions:
        if p.get("category_id", -1) != -1:
            pred_by_img[p["image_id"]].append(p)

    for img_id in img_ids:
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        gt_anns = coco_gt.loadAnns(ann_ids)   # list of {category_id, bbox, ...}

        preds = sorted(
            pred_by_img.get(img_id, []),
            key=lambda p: p.get("score", 0.0),
            reverse=True,
        )

        gt_claimed = [False] * len(gt_anns)

        for pred in preds:
            pred_idx = catid2idx.get(pred["category_id"], -1)
            if pred_idx == -1:
                continue

            pred_bbox = pred["bbox"]

            # Find best-IoU unclaimed GT box
            best_iou  = 0.0
            best_gt_i = -1
            for gi, gt in enumerate(gt_anns):
                if gt_claimed[gi]:
                    continue
                iou = compute_iou(pred_bbox, gt["bbox"])
                if iou > best_iou:
                    best_iou  = iou
                    best_gt_i = gi

            if best_iou >= iou_threshold and best_gt_i != -1:
                # Prediction is localised — claim the GT box
                gt_claimed[best_gt_i] = True
                gt_cat_idx = catid2idx.get(gt_anns[best_gt_i]["category_id"], -1)
                if gt_cat_idx != -1:
                    cm[pred_idx, gt_cat_idx] += 1
            else:
                # Unmatched prediction → background FP column
                cm[pred_idx, BKG] += 1

        # Unclaimed GT boxes → background FN row
        for gi, gt in enumerate(gt_anns):
            if not gt_claimed[gi]:
                gt_cat_idx = catid2idx.get(gt["category_id"], -1)
                if gt_cat_idx != -1:
                    cm[BKG, gt_cat_idx] += 1

    labels = cat_names + ["background"]
    return cm, labels


# ---------------------------------------------------------------------------
# Confusion matrix plotting
# ---------------------------------------------------------------------------

def _pick_font_size(n: int) -> int:
    """Scale annotation font size with matrix dimension."""
    if n <= 4:  return 12
    if n <= 8:  return 10
    if n <= 14: return 8
    return 6


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list,
    dataset_name: str,
    out_path: str,
    normalize: bool = True,
    iou_threshold: float = 0.5,
):
    """
    Plot and save a single confusion matrix heatmap.

    Parameters
    ----------
    cm           : raw count matrix (n+1, n+1)
    labels       : class names including 'background' as last entry
    dataset_name : used in the title
    out_path     : full path for the saved PNG
    normalize    : if True, normalise each GT column (col-wise recall view)
    iou_threshold: shown in the title for reference
    """
    n = len(labels)

    # Column-wise normalisation: each column sums to 1 (GT-class recall view).
    # Rows show what fraction of GT-class j was predicted as each class i.
    if normalize:
        col_sums = cm.sum(axis=0, keepdims=True).astype(float)
        col_sums[col_sums == 0] = 1          # avoid /0
        cm_plot = cm.astype(float) / col_sums
        fmt     = ".2f"
        cb_label = "Fraction of GT column"
    else:
        cm_plot = cm.astype(float)
        fmt     = ".0f"
        cb_label = "Count"

    # Mask exact zeros so the heatmap stays readable
    mask = (cm == 0)

    fig_size = max(6, n * 0.7)
    fig, ax  = plt.subplots(figsize=(fig_size + 1.5, fig_size), dpi=120)

    sns.heatmap(
        cm_plot,
        mask=mask,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.4,
        linecolor="#cccccc",
        cbar_kws={"label": cb_label, "shrink": 0.7},
        ax=ax,
        annot_kws={"size": _pick_font_size(n)},
        vmin=0,
        vmax=1 if normalize else None,
    )

    ax.set_xlabel("Ground-Truth Class", fontsize=11, labelpad=8)
    ax.set_ylabel("Predicted Class",    fontsize=11, labelpad=8)
    ax.set_title(
        f"{dataset_name}\nConfusion matrix  (IoU ≥ {iou_threshold},"
        f"  {'column-normalised' if normalize else 'raw counts'})",
        fontsize=11, pad=12,
    )

    # Highlight the diagonal (TP) with a subtle box
    for i in range(n - 1):   # skip background diagonal
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                                   edgecolor="#e74c3c", lw=1.2))

    # Rotate tick labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  [CM] Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Aggregate confusion matrix (sum across datasets, then re-normalise)
# ---------------------------------------------------------------------------

def plot_aggregate_confusion_matrix(
    cms_by_dataset: dict[str, tuple[np.ndarray, list]],
    out_path: str,
    model_name: str,
    iou_threshold: float = 0.5,
):
    """
    Sum all per-dataset confusion matrices that share the same label set,
    then plot the aggregate normalised matrix.

    Datasets with different label sets (different number of classes) are
    plotted separately in the per-dataset step and excluded from the aggregate.
    """
    # Group datasets by their label tuple so we only sum compatible matrices
    groups: dict[tuple, list] = defaultdict(list)
    for ds_name, (cm, labels) in cms_by_dataset.items():
        key = tuple(labels)
        groups[key].append((ds_name, cm, labels))

    for label_tuple, members in groups.items():
        labels  = list(label_tuple)
        agg_cm  = sum(cm for _, cm, _ in members)
        ds_list = ", ".join(ds for ds, _, _ in members)

        title_extra = (
            f"Aggregate over {len(members)} dataset(s)\n"
            f"({ds_list[:80]}{'...' if len(ds_list) > 80 else ''})"
        )

        # Save alongside individual plots
        stem     = out_path.replace(".png", f"_n{len(members)}.png")
        n        = len(labels)
        fig_size = max(6, n * 0.7)
        fig, ax  = plt.subplots(figsize=(fig_size + 1.5, fig_size), dpi=120)

        col_sums = agg_cm.sum(axis=0, keepdims=True).astype(float)
        col_sums[col_sums == 0] = 1
        cm_norm  = agg_cm.astype(float) / col_sums
        mask     = (agg_cm == 0)

        sns.heatmap(
            cm_norm, mask=mask, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            linewidths=0.4, linecolor="#cccccc",
            cbar_kws={"label": "Fraction of GT column", "shrink": 0.7},
            ax=ax, annot_kws={"size": _pick_font_size(n)},
            vmin=0, vmax=1,
        )
        for i in range(n - 1):
            ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                                       edgecolor="#e74c3c", lw=1.2))

        ax.set_xlabel("Ground-Truth Class", fontsize=11, labelpad=8)
        ax.set_ylabel("Predicted Class",    fontsize=11, labelpad=8)
        ax.set_title(
            f"{model_name} — {title_extra}\n"
            f"Confusion matrix  (IoU ≥ {iou_threshold}, column-normalised)",
            fontsize=10, pad=12,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=8)

        plt.tight_layout()
        plt.savefig(stem, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"[CM] Aggregate plot saved -> {stem}")


# ---------------------------------------------------------------------------
# Per-dataset TP / FP / FN summary bar chart
# ---------------------------------------------------------------------------

def plot_cm_summary_bars(
    cms_by_dataset: dict[str, tuple[np.ndarray, list]],
    out_path: str,
    model_name: str,
):
    """
    For each dataset compute overall TP, FP (localised wrong class),
    FP-background (unmatched pred), and FN (missed GT), then plot a
    stacked bar chart so you can compare datasets at a glance.
    """
    rows = []
    for ds_name, (cm, labels) in cms_by_dataset.items():
        n   = len(labels) - 1    # exclude background pseudo-class
        BKG = n

        tp       = int(np.trace(cm[:n, :n]))               # diagonal, no bkg
        fp_cls   = int(cm[:n, :n].sum() - tp)              # off-diagonal, localised
        fp_bkg   = int(cm[:n, BKG].sum())                  # pred with no GT match
        fn       = int(cm[BKG, :n].sum())                  # missed GT boxes
        total_gt = tp + fp_cls + fn                        # all localised + FN

        rows.append({
            "Dataset": ds_name,
            "TP":           tp,
            "FP (cls err)": fp_cls,
            "FP (no match)": fp_bkg,
            "FN":           fn,
            "Precision": tp / max(tp + fp_cls + fp_bkg, 1),
            "Recall":    tp / max(tp + fn, 1),
        })

    df = pd.DataFrame(rows).set_index("Dataset")

    # --- Stacked bar of TP / FP-cls / FP-bkg / FN ---
    fig, axes = plt.subplots(1, 2, figsize=(18, max(5, len(rows) * 0.45 + 2)), dpi=120)
    fig.suptitle(f"{model_name} — detection breakdown per dataset", fontsize=13,
                 fontweight="bold", y=1.01)

    colors = {"TP": "#2ecc71", "FP (cls err)": "#e67e22",
              "FP (no match)": "#e74c3c", "FN": "#95a5a6"}

    count_cols = ["TP", "FP (cls err)", "FP (no match)", "FN"]
    df[count_cols].plot(
        kind="barh", stacked=True, ax=axes[0],
        color=[colors[c] for c in count_cols],
        edgecolor="white", linewidth=0.4,
    )
    axes[0].set_xlabel("Box count", fontsize=10)
    axes[0].set_ylabel("")
    axes[0].set_title("Absolute counts", fontsize=11)
    axes[0].legend(loc="lower right", fontsize=8)
    sns.despine(ax=axes[0], left=True, bottom=True)

    # --- Precision / Recall scatter ---
    ax2 = axes[1]
    scatter_colors = plt.cm.tab20(np.linspace(0, 1, len(rows)))
    for i, (idx, row) in enumerate(df.iterrows()):
        ax2.scatter(row["Recall"], row["Precision"], color=scatter_colors[i],
                    s=80, zorder=3, label=idx)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel("Recall  (TP / (TP + FN))", fontsize=10)
    ax2.set_ylabel("Precision  (TP / (TP + FP))", fontsize=10)
    ax2.set_title("Precision vs Recall per dataset", fontsize=11)
    ax2.axline((0, 0), slope=1, color="#bdc3c7", ls="--", lw=0.8)
    ax2.legend(fontsize=6, bbox_to_anchor=(1.01, 1), loc="upper left",
               borderaxespad=0, ncol=1)
    ax2.grid(True, alpha=0.3)
    sns.despine(ax=ax2)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[CM] Summary bar chart saved -> {out_path}")

    # Also save the numbers as CSV
    csv_path = out_path.replace(".png", ".csv")
    df.reset_index().to_csv(csv_path, index=False)
    print(f"[CM] Summary CSV saved -> {csv_path}")


# ---------------------------------------------------------------------------
# Core per-dataset evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(
    tide,
    dataset_name: str,
    dataset_root: str,
    results_root: str,
    data_instr_path: str,
    score_type: str = "rank",
    iou_threshold: float = 0.5,
    output_dir: str = None,
):
    """
    For one dataset:
      • locate annotations and predictions
      • run COCO eval
      • run TIDE eval
      • compute confusion matrix

    Returns
    -------
    coco_stats : dict  { score_type: [12 floats] }   or None on failure
    cm_result  : tuple (cm_array, labels)             or None on failure
    """
    # --- Locate ground truth annotations ---
    dataset_path = os.path.join(dataset_root, dataset_name)
    if not os.path.isdir(dataset_path):
        candidates = glob.glob(os.path.join(dataset_root, f"{dataset_name}*"))
        if not candidates:
            print(f"[SKIP] Dataset directory not found for '{dataset_name}'")
            return None, None
        dataset_path = candidates[0]
        print(f"[INFO] Using dataset path: {dataset_path}")

    ann_path = os.path.join(dataset_path, "test", "_annotations.coco.json")
    if not os.path.isfile(ann_path):
        print(f"[SKIP] No test annotations at {ann_path}")
        return None, None

    coco_gt = COCO(ann_path)

    # --- Locate predictions JSON ---
    subfolder = "rank" if score_type == "model" else score_type
    predictions_path = os.path.join(
        results_root, "final_instruction_eval", "predictions",
        subfolder, f"predictions_{dataset_name}_{score_type}.json"
    )
    if not os.path.isfile(predictions_path):
        found = glob.glob(
            os.path.join(results_root, "**",
                         f"predictions_{dataset_name}_{score_type}.json"),
            recursive=True,
        )
        if not found:
            print(f"[SKIP] predictions not found for '{dataset_name}'")
            return None, None
        predictions_path = found[0]
        print(f"[INFO] Found predictions at: {predictions_path}")

    with open(predictions_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    valid = [d for d in predictions if d.get("category_id", -1) != -1]

    # ---- COCO evaluation ----
    coco_stats = {}
    if not valid:
        print(f"  [WARN] No valid predictions for '{score_type}'")
        coco_stats[score_type] = [0.0] * 12
    else:
        try:
            coco_dt   = coco_gt.loadRes(valid)
            coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            coco_stats[score_type] = coco_eval.stats.tolist()
        except Exception as e:
            print(f"  [ERROR] COCO eval failed: {e}")
            coco_stats[score_type] = [0.0] * 12

    # ---- TIDE evaluation ----
    try:
        tide_ds = datasets.COCO(ann_path)
        tide_dt = datasets.COCOResult(predictions_path)
        tide.evaluate(tide_ds, tide_dt, mode=TIDE.BOX,
                      name=dataset_name, use_for_errors=True)
    except Exception as e:
        print(f"  [ERROR] TIDE analysis failed: {e}")

    # ---- Confusion matrix ----
    cm_result = None
    try:
        cm, labels = compute_confusion_matrix(
            coco_gt, valid, iou_threshold=iou_threshold
        )
        cm_result = (cm, labels)

        # Print per-class TP / FP / FN to terminal
        n   = len(labels) - 1
        BKG = n
        print(f"  [CM] IoU≥{iou_threshold} | classes: {labels[:-1]}")
        print(f"       {'Class':<24} {'TP':>6} {'FP-cls':>8} {'FP-bkg':>8} {'FN':>6}")
        for i, cls in enumerate(labels[:-1]):
            tp     = int(cm[i, i])
            fp_cls = int(cm[i, :n].sum()) - tp
            fp_bkg = int(cm[i, BKG])
            fn     = int(cm[BKG, i])
            print(f"       {cls:<24} {tp:>6} {fp_cls:>8} {fp_bkg:>8} {fn:>6}")

    except Exception as e:
        print(f"  [ERROR] Confusion matrix computation failed: {e}")

    return coco_stats, cm_result


# ---------------------------------------------------------------------------
# TIDE error consolidation
# ---------------------------------------------------------------------------

def consolidate_tide_errors(errors_per_dataset: dict) -> dict:
    """Average main and special TIDE errors across all datasets."""
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
    """Produce consolidated TIDE summary plot + per-dataset spread plot."""
    os.makedirs(out_dir, exist_ok=True)
    consolidated = consolidate_tide_errors(errors_per_dataset)
    n_datasets   = len(errors_per_dataset.get("main", {}))

    print(f"\n[TIDE] Consolidated errors across {n_datasets} datasets:")
    for k, v in consolidated["main"]["consolidated"].items():
        print(f"  {k:10s}: {v:.3f}")
    for k, v in consolidated["special"]["consolidated"].items():
        print(f"  {k:12s}: {v:.3f}")

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

    src = os.path.join(out_dir, f"{model_name}_{rec_type}_summary.png")
    dst = os.path.join(out_dir, f"{model_name}_{rec_type}_consolidated_summary.png")
    if os.path.isfile(src) and src != dst:
        os.replace(src, dst)
    print(f"[TIDE] Saved consolidated summary -> {dst}")

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
    errors_per_dataset, out_dir, model_name, rec_type,
    colors_main, colors_special, n_datasets,
):
    main_ds    = errors_per_dataset.get("main",    {})
    special_ds = errors_per_dataset.get("special", {})

    empty_cols = ["Dataset", "Error Type", "Delta mAP"]
    main_rows = [
        {"Dataset": ds, "Error Type": err, "Delta mAP": val}
        for ds, err_dict in main_ds.items() for err, val in err_dict.items()
    ]
    special_rows = [
        {"Dataset": ds, "Error Type": err, "Delta mAP": val}
        for ds, err_dict in special_ds.items() for err, val in err_dict.items()
    ]
    main_df    = pd.DataFrame(main_rows)    if main_rows    else pd.DataFrame(columns=empty_cols)
    special_df = pd.DataFrame(special_rows) if special_rows else pd.DataFrame(columns=empty_cols)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150,
                             gridspec_kw={"width_ratios": [3, 1]})
    fig.suptitle(f"{model_name} — TIDE error breakdown across {n_datasets} datasets",
                 fontsize=13, fontweight="bold", y=1.02)

    ax_main = axes[0]
    if not main_df.empty:
        error_order = list(colors_main.keys())
        sns.barplot(data=main_df, x="Delta mAP", y="Error Type",
                    order=error_order, palette=dict(colors_main),
                    estimator=np.mean, errorbar="sd",
                    ax=ax_main, capsize=0.3, errwidth=1.5)
        sns.stripplot(data=main_df, x="Delta mAP", y="Error Type",
                      order=error_order,
                      color="black", alpha=0.4, size=4, jitter=False, ax=ax_main)
        mean_vals = main_df.groupby("Error Type")["Delta mAP"].mean()
        for i, err_type in enumerate(error_order):
            if err_type in mean_vals:
                ax_main.text(mean_vals[err_type] + 0.15, i,
                             f"{mean_vals[err_type]:.2f}",
                             va="center", fontsize=9, color="#333333")
    ax_main.set_xlabel("Δ mAP", fontsize=11)
    ax_main.set_ylabel("")
    ax_main.set_title("Main error types (mean ± std)", fontsize=11)
    sns.despine(ax=ax_main, left=True, bottom=True)

    ax_spec = axes[1]
    if not special_df.empty:
        error_order_s = list(colors_special.keys())
        sns.barplot(data=special_df, x="Error Type", y="Delta mAP",
                    order=error_order_s, palette=dict(colors_special),
                    estimator=np.mean, errorbar="sd",
                    ax=ax_spec, capsize=0.3, errwidth=1.5)
        sns.stripplot(data=special_df, x="Error Type", y="Delta mAP",
                      order=error_order_s,
                      color="black", alpha=0.4, size=4, jitter=False, ax=ax_spec)
        mean_vals_s = special_df.groupby("Error Type")["Delta mAP"].mean()
        for i, err_type in enumerate(error_order_s):
            if err_type in mean_vals_s:
                ax_spec.text(i, mean_vals_s[err_type] + 0.3,
                             f"{mean_vals_s[err_type]:.2f}",
                             ha="center", fontsize=9, color="#333333")
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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2)
    print(f"[TIDE] Errors saved -> {path}")


def load_tide_errors(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# COCO metric consolidation
# ---------------------------------------------------------------------------

def consolidate(all_dataset_stats: dict, output_dir: str):
    """Save per-eval-type CSVs + summary.txt from COCO stats."""
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

        df  = pd.DataFrame(records)
        avg = df.drop(columns=["dataset"]).mean()
        df  = pd.concat([df, pd.DataFrame([{"dataset": "AVERAGE", **avg.to_dict()}])],
                        ignore_index=True)
        csv_path = os.path.join(output_dir, f"results_{eval_type}.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n[{eval_type}] Saved: {csv_path}")

        cat_groups: dict[str, list] = defaultdict(list)
        for row in records:
            cat_groups[get_category(row["dataset"])].append(row["AP_50_95"])

        block = [f"\n=== {eval_type} ===",
                 f"  Overall mAP (AP50-95): {avg['AP_50_95']*100:.2f}"]
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
    parser = argparse.ArgumentParser(
        description="Evaluate all 20 datasets: COCO metrics, TIDE errors, confusion matrices."
    )
    parser.add_argument(
        "--results_root", type=str,
        default="results/results_data3/final_consolidated_results/rf-20-vl-benchmark/"
                "results/eccv26/rf100vl_IPT/Qwen3-VL-30B-A3B-Instruct/rf20_IPT_singleclass_rankScore",
    )
    parser.add_argument("--dataset_root",    type=str, default="./datasets/rf100-vl-fsod")
    parser.add_argument(
        "--data_instr_path", type=str,
        default="results/rf100vl_IPT/Qwen2.5-VL-7B-Instruct/"
                "rf20_IPT_singleclass_vqaScore_withNMS/iterative_prompt_refinement/"
                "all_refined_class_instructions",
    )
    parser.add_argument("--score_type",       type=str,  
                        # default="model",
                        default="vqa",
                        help="'model', 'rank', etc.")
    parser.add_argument("--iou_threshold",    type=float, default=0.5,
                        help="IoU threshold for confusion matrix localisation filter.")
    parser.add_argument("--nms_iou_threshold",type=float, default=0.1)
    parser.add_argument("--output_dir",       type=str,  default=None)
    parser.add_argument("--model_name",       type=str,  default=None)
    parser.add_argument("--no_normalize_cm",  action="store_true",
                        help="Plot raw counts instead of column-normalised confusion matrices.")
    args = parser.parse_args()

    # output_dir = args.output_dir or os.path.join(args.results_root, "analyse_detection_errors")
    output_dir = args.output_dir or os.path.join(args.results_root, "analyse_confusion_matrices", args.score_type)
    os.makedirs(output_dir, exist_ok=True)

    model_name = args.model_name or Path(args.results_root).parts[-2]
    normalize_cm = not args.no_normalize_cm

    cm_out_dir = os.path.join(output_dir, "confusion_matrices")
    os.makedirs(cm_out_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Per-dataset evaluation                                           #
    # ------------------------------------------------------------------ #
    tide = TIDE()
    all_dataset_stats:  dict[str, dict]                    = {}
    cms_by_dataset:     dict[str, tuple[np.ndarray, list]] = {}

    for dataset_name in DATASETS:
        print(f"\n{'='*60}")
        print(f"Processing: {dataset_name}")

        coco_stats, cm_result = evaluate_dataset(
            tide=tide,
            dataset_name=dataset_name,
            dataset_root=args.dataset_root,
            results_root=args.results_root,
            data_instr_path=args.data_instr_path,
            score_type=args.score_type,
            iou_threshold=args.iou_threshold,
            output_dir=output_dir,
        )

        if coco_stats is not None:
            all_dataset_stats[dataset_name] = coco_stats
            for eval_type, s in coco_stats.items():
                print(f"  [{eval_type}] AP50-95={s[0]*100:.2f}  "
                      f"AP50={s[1]*100:.2f}  AR@1={s[6]*100:.2f}")

        if cm_result is not None:
            cm, labels = cm_result
            cms_by_dataset[dataset_name] = cm_result

            # Plot per-dataset confusion matrix
            cm_path = os.path.join(cm_out_dir, f"{dataset_name}_confusion_matrix.png")
            plot_confusion_matrix(
                cm=cm,
                labels=labels,
                dataset_name=dataset_name,
                out_path=cm_path,
                normalize=normalize_cm,
                iou_threshold=args.iou_threshold,
            )

    print(f"\n{'='*60}")
    print(f"Processed {len(all_dataset_stats)} / {len(DATASETS)} datasets.")

    # ------------------------------------------------------------------ #
    # 2. COCO metric consolidation                                        #
    # ------------------------------------------------------------------ #
    consolidate(all_dataset_stats, output_dir)

    # # ------------------------------------------------------------------ #
    # # 3. TIDE: terminal summary + per-dataset plots                       #
    # # ------------------------------------------------------------------ #
    # print(f"\n{'='*60}")
    # print("TIDE summary across all datasets:")
    # tide.summarize()

    # tide_plots_dir = os.path.join(output_dir, "tide_plots_per_dataset")
    # tide.plot(out_dir=tide_plots_dir)
    # print(f"[TIDE] Per-dataset plots saved -> {tide_plots_dir}")

    # # ------------------------------------------------------------------ #
    # # 4. Consolidated TIDE error analysis + plots                         #
    # # ------------------------------------------------------------------ #
    # tide_errors      = tide.get_all_errors()
    # errors_json_path = os.path.join(output_dir, f"tide_errors_{args.score_type}.json")
    # save_tide_errors(tide_errors, errors_json_path)

    # consolidated_errors = plot_consolidated_tide_errors(
    #     errors_per_dataset=tide_errors,
    #     out_dir=os.path.join(output_dir, "tide_plots_consolidated"),
    #     model_name=model_name,
    #     rec_type="bbox",
    # )

    # ------------------------------------------------------------------ #
    # 5. Aggregate confusion matrix + summary bar chart                   #
    # ------------------------------------------------------------------ #
    if cms_by_dataset:
        plot_aggregate_confusion_matrix(
            cms_by_dataset=cms_by_dataset,
            out_path=os.path.join(cm_out_dir, f"{model_name}_aggregate_confusion_matrix.png"),
            model_name=model_name,
            iou_threshold=args.iou_threshold,
        )
        plot_cm_summary_bars(
            cms_by_dataset=cms_by_dataset,
            out_path=os.path.join(cm_out_dir, f"{model_name}_cm_summary_bars.png"),
            model_name=model_name,
        )

    # # ------------------------------------------------------------------ #
    # # 6. Print consolidated TIDE error table                              #
    # # ------------------------------------------------------------------ #
    # print(f"\n{'='*60}")
    # print(f"Consolidated TIDE errors for {model_name}:")
    # all_errors = {
    #     **consolidated_errors["main"]["consolidated"],
    #     **consolidated_errors["special"]["consolidated"],
    # }
    # col_w = [max(len("Error Type"), max(len(k) for k in all_errors)), 10]
    # div   = "  " + "---".join("-" * w for w in col_w)
    # print(div.replace("-", "="))
    # print("  " + "   ".join([f"{'Error Type':>{col_w[0]}}", f"{'Mean Δ mAP':>{col_w[1]}}"]))
    # print(div)
    # for k, v in all_errors.items():
    #     print("  " + "   ".join([f"{k:>{col_w[0]}}", f"{v:>{col_w[1]}.3f}"]))
    # print(div.replace("-", "="))


if __name__ == "__main__":
    main()