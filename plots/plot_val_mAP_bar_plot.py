import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
from collections import defaultdict
from rf100vl.util import get_category

# ── Config ────────────────────────────────────────────────────────────────────
base_dir   = (
    "results/final_consolidated_results/rf-20-vl-benchmark/results/eccv26/"
    "rf100vl_IPT/Qwen3-VL-30B-A3B-Instruct/rf20_IPT_singleclass_rankScore/"
    "iterative_prompt_refinement"
)
METRIC_KEY = "mAP_50_95"

# Instruction types to compare (fixed) + "best of all" computed dynamically
FIXED_TYPES = ["original_instructions", "initial_instructions"]

category_renames = {"Aerial": "Aerial", "Lab Imaging": "Medical", "Sport": "Sports", "Document": "Document", "Flora/Fauna": "Flora & Fauna", "Misc": "Others", "Industrial": "Industrial"}

# ── Data loading ───────────────────────────────────────────────────────────────
dataset_dirs = sorted(glob.glob(os.path.join(base_dir, "*")))

# category → {instr_type: [per-class mAP values]}
category_data = defaultdict(lambda: defaultdict(list))

for dataset_dir in dataset_dirs:
    dataset_name = os.path.basename(dataset_dir)
    files = glob.glob(
        os.path.join(
            dataset_dir,
            f"valSet_instruction_eval_results_{dataset_name}_cls_*.json",
        )
    )
    if not files:
        continue

    category = get_category(dataset_name)

    for file in files:
        with open(file) as f:
            data = json.load(f)

        cls_name = file.split("_cls_")[-1].replace(".json", "")

        # Fixed types
        for instr_type in FIXED_TYPES:
            key = f"class_{cls_name}_{instr_type}"
            if key in data:
                category_data[category][instr_type].append(
                    data[key][METRIC_KEY]
                )

        # Best across ALL instruction types present in this file
        all_maps = [
            v[METRIC_KEY]
            for v in data.values()
            if isinstance(v, dict) and METRIC_KEY in v
        ]
        if all_maps:
            category_data[category]["best_any_instruction"].append(max(all_maps))

# ── Aggregate: mean mAP per category per instruction type ────────────────────
INSTR_LABELS = {
    "original_instructions":  "Original",
    "initial_instructions":   "DetPO Initial",
    "best_any_instruction":   "DetPO Optimized",
}
instr_keys   = list(INSTR_LABELS.keys())
categories   = sorted(category_data.keys())
n_cats       = len(categories)
n_bars       = len(instr_keys)

means = {k: [] for k in instr_keys}
sems  = {k: [] for k in instr_keys}

for cat in categories:
    for k in instr_keys:
        vals = category_data[cat].get(k, [])
        means[k].append(np.nanmean(vals) if vals else np.nan)
        sems[k].append(
            np.nanstd(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        )

# ── Seaborn theme ─────────────────────────────────────────────────────────────
sns.set_theme(
    style="whitegrid",
    context="paper",
    font="DejaVu Sans",
    rc={
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "grid.color":        "#e5e7eb",
        "grid.linewidth":    0.8,
        "axes.facecolor":    "#fafafa",
        "figure.facecolor":  "white",
    },
)

palette = sns.color_palette("Set2", n_colors=n_bars)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(max(10, n_cats * 1.6), 6))

bar_width = 0.25
offsets   = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * bar_width
x         = np.arange(n_cats)

for i, (k, color) in enumerate(zip(instr_keys, palette)):
    bar_x  = x + offsets[i]
    bar_y  = np.array(means[k], dtype=float)
    err    = np.array(sems[k],  dtype=float)

    bars = ax.bar(
        bar_x, bar_y, bar_width,
        color=color, alpha=0.88,
        label=INSTR_LABELS[k],
        zorder=3,
    )
    ax.errorbar(
        bar_x, bar_y, yerr=err,
        fmt="none", color="#374151",
        capsize=3, linewidth=1.1, zorder=4,
    )
    # value labels on top of each bar (clear of the error cap)
    for bx, by, be in zip(bar_x, bar_y, err):
        if not np.isnan(by):
            ax.text(
                bx, by + be + 0.004,
                f"{by:.3f}",
                ha="center", va="bottom",
                fontsize=6.5, color="#374151",
            )

# ── Annotations: dataset count per category ───────────────────────────────────
for ci, cat in enumerate(categories):
    n_ds = len(set(
        # count datasets (files may have multiple classes)
        # approximate via number of "original" values as proxy
        category_data[cat].get("original_instructions", [])
    ))
    ax.text(
        x[ci], -0.012,
        f"n={len(category_data[cat].get('original_instructions', []))} cls",
        ha="center", va="top",
        fontsize=7, color="#6b7280",
        transform=ax.get_xaxis_transform(),
    )

# ── Labels & formatting ───────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels([category_renames.get(cat, cat) for cat in categories], rotation=25, ha="right", fontsize=10)
ax.set_ylabel(f"Mean {METRIC_KEY.replace('_', ' ')} (val set)", fontsize=12, labelpad=8)
ax.set_title(
    "Instruction Type Comparison by Dataset Category",
    fontsize=13, fontweight="bold", pad=14,
)
ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.3f"))
ax.tick_params(axis="y", labelsize=9)

legend = ax.legend(
    title="Instruction type",
    title_fontsize=10,
    fontsize=9,
    loc="upper right",
    frameon=True,
    framealpha=0.9,
    edgecolor="#d1d5db",
)
legend.get_frame().set_linewidth(0.8)

plt.tight_layout()
plt.savefig(
    "instruction_type_comparison_by_category.png",
    dpi=300, bbox_inches="tight",
)
plt.show()

# ── Console summary ───────────────────────────────────────────────────────────
print(f"\n{'Category':<28} {'Original':>10} {'Initial':>10} {'Best':>10}  {'Δ Best-Orig':>12}")
print("-" * 74)
for ci, cat in enumerate(categories):
    orig = means["original_instructions"][ci]
    init = means["initial_instructions"][ci]
    best = means["best_any_instruction"][ci]
    delta = best - orig if not (np.isnan(best) or np.isnan(orig)) else float("nan")
    print(f"{cat:<28} {orig:>10.4f} {init:>10.4f} {best:>10.4f}  {delta:>+12.4f}")