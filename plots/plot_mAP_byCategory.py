import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import os
from collections import defaultdict
from rf100vl.util import get_category

# ── Config ──────────────────────────────────────────────────────────────────
base_dir = (
    "results/final_consolidated_results/rf-20-vl-benchmark/results/eccv26/"
    "rf100vl_IPT/Qwen3-VL-30B-A3B-Instruct/rf20_IPT_singleclass_rankScore/"
    "iterative_prompt_refinement"
)
METRIC_KEY = "mAP_50_95"
num_iters  = 11

category_renames = {"Aerial": "Aerial", "Lab Imaging": "Medical", "Sport": "Sports", "Document": "Document", "Flora/Fauna": "Flora & Fauna", "Misc": "Others", "Industrial": "Industrial"}

# ── Data loading ─────────────────────────────────────────────────────────────
dataset_dirs    = sorted(glob.glob(os.path.join(base_dir, "*")))
category_curves = defaultdict(list)

for dataset_dir in dataset_dirs:
    dataset_name = os.path.basename(dataset_dir)
    files = glob.glob(
        os.path.join(
            dataset_dir,
            f"instruction_refinements_log_{dataset_name}_cls_*.json"
        )
    )
    if not files:
        continue

    all_results = []
    for file in files:
        cls_name = file.split("cls_")[-1].replace(".json", "")
        with open(file) as f:
            data = json.load(f)

        iter_values = [
            data.get(f"class_{cls_name}_iter_{i}", {}).get(METRIC_KEY, np.nan)
            for i in range(num_iters)
        ]
        all_results.append(np.maximum.accumulate(iter_values))

    all_results = np.array(all_results)
    mean_map    = np.nanmean(all_results, axis=0)
    base        = mean_map[0]
    improvement_curve = (mean_map - base) * 100

    category = get_category(dataset_name)
    category_curves[category].append(improvement_curve)

# ── Seaborn theme ─────────────────────────────────────────────────────────────
sns.set_theme(
    style="whitegrid",
    context="paper",
    font="DejaVu Sans",
    rc={
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "grid.color":         "#e5e7eb",
        "grid.linewidth":     0.8,
        "axes.facecolor":     "#fafafa",
        "figure.facecolor":   "white",
    },
)

palette = sns.color_palette("tab10", n_colors=len(category_curves))
iterations = np.arange(num_iters)

fig, ax = plt.subplots(figsize=(11, 6.5))

# ── Plot each category ────────────────────────────────────────────────────────
for idx, (category, curves) in enumerate(
    sorted(category_curves.items(), key=lambda kv: -np.nanmean(kv[1]))
):
    curves     = np.array(curves)
    mean_curve = np.nanmean(curves, axis=0)
    std_curve  = np.nanstd(curves,  axis=0)
    color      = palette[idx]

    # ax.fill_between(
    #     iterations,
    #     mean_curve - std_curve,
    #     mean_curve + std_curve,
    #     alpha=0.12,
    #     color=color,
    #     linewidth=0,
    # )
    ax.plot(
        iterations,
        mean_curve,
        marker="o",
        markersize=5,
        linewidth=2.2,
        # label=f"{category}  (n={len(curves)})",
        label=f"{category_renames[category]}",
        color=color,
        solid_capstyle="round",
        solid_joinstyle="round",
    )
    # annotate final value on the right
    ax.annotate(
        f"{mean_curve[-1]:+.1f}",
        xy=(iterations[-1], mean_curve[-1]),
        xytext=(4, 0),
        textcoords="offset points",
        fontsize=7.5,
        color=color,
        va="center",
        fontweight="semibold",
    )

# ── Baseline ──────────────────────────────────────────────────────────────────
ax.axhline(0, linestyle="--", linewidth=1.2, color="#6b7280", zorder=1)

# ── Axes & labels ─────────────────────────────────────────────────────────────
ax.set_xlabel("Refinement Iteration", fontsize=12, labelpad=8)
ax.set_ylabel("Δ mAP\u2080.₅:₀.₉₅ vs. Iter 0  (pp)", fontsize=12, labelpad=8)
ax.set_title(
    "DetPO Improvement by Dataset Category",
    fontsize=13.5,
    fontweight="bold",
    pad=14,
)

ax.set_xticks(iterations)
ax.xaxis.set_minor_locator(ticker.NullLocator())
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%+.1f"))
ax.tick_params(axis="both", labelsize=10)

# ── Legend ────────────────────────────────────────────────────────────────────
legend = ax.legend(
    title="Category",
    title_fontsize=10,
    fontsize=9,
    bbox_to_anchor=(1.01, 1),
    loc="upper left",
    borderaxespad=0,
    frameon=True,
    framealpha=0.9,
    edgecolor="#d1d5db",
)
legend.get_frame().set_linewidth(0.8)

plt.tight_layout()
plt.savefig(
    "instruction_refinement_category_improvement.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# ── Summary table ─────────────────────────────────────────────────────────────
print("\nCategory Summary  (sorted by final improvement):")
print(f"{'Category':<30} {'Mean Δ mAP (pp)':>16}  {'# datasets':>11}")
print("-" * 62)
records = []
for category, curves in category_curves.items():
    curves = np.array(curves)
    final  = curves[:, -1]
    records.append(
        dict(category=category, mean_improvement=np.mean(final), n=len(curves))
    )
for r in sorted(records, key=lambda x: -x["mean_improvement"]):
    print(f"{r['category']:<30} {r['mean_improvement']:>+15.2f}  {r['n']:>11}")