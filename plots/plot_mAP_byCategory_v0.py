import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict

from rf100vl.util import get_category

base_dir = "results/final_consolidated_results/rf-20-vl-benchmark/results/eccv26/rf100vl_IPT/Qwen3-VL-30B-A3B-Instruct/rf20_IPT_singleclass_rankScore/iterative_prompt_refinement"

METRIC_KEY = "mAP_50_95"
num_iters = 11

dataset_dirs = sorted(glob.glob(os.path.join(base_dir, "*")))

# store dataset curves grouped by category
category_curves = defaultdict(list)

for dataset_dir in dataset_dirs:

    dataset_name = os.path.basename(dataset_dir)

    files = glob.glob(
        os.path.join(dataset_dir, f"instruction_refinements_log_{dataset_name}_cls_*.json")
    )

    if len(files) == 0:
        continue

    all_results = []

    for file in files:

        cls_name = file.split('cls_')[-1].replace('.json','')

        with open(file, "r") as f:
            data = json.load(f)

        iter_values = []

        for i in range(num_iters):

            key = f"class_{cls_name}_iter_{i}"

            if key in data:
                iter_values.append(data[key][METRIC_KEY])
            else:
                iter_values.append(np.nan)

        # best-so-far curve
        iter_values = np.maximum.accumulate(iter_values)

        all_results.append(iter_values)

    all_results = np.array(all_results)

    # average across classes
    mean_map = np.nanmean(all_results, axis=0)

    # absolute improvement vs iter0
    base = mean_map[0]
    improvement_curve = (mean_map - base) * 100

    # dataset category
    category = get_category(dataset_name)

    category_curves[category].append(improvement_curve)

# ===============================
# Plot Category Curves
# ===============================

plt.figure(figsize=(10,6))

colors = plt.cm.tab10(np.linspace(0,1,len(category_curves)))

for idx, (category, curves) in enumerate(category_curves.items()):

    curves = np.array(curves)

    mean_curve = np.nanmean(curves, axis=0)
    std_curve = np.nanstd(curves, axis=0)

    iterations = np.arange(num_iters)

    plt.plot(
        iterations,
        mean_curve,
        marker='o',
        linewidth=2.5,
        # label=f"{category} (n={len(curves)})",
        label=f"{category}",
        color=colors[idx]
    )

    # plt.fill_between(
    #     iterations,
    #     mean_curve - std_curve,
    #     mean_curve + std_curve,
    #     alpha=0.2,
    #     color=colors[idx]
    # )

plt.axhline(0, linestyle="--", color="black")

plt.xlabel("Iteration", fontsize=13)
plt.ylabel("Absolute Improvement over Iter 0 (mAP points)", fontsize=13)
plt.title("Instruction Refinement Improvement by Dataset Category", fontsize=14)

plt.xticks(np.arange(num_iters))
plt.grid(True, linestyle="--", alpha=0.4)

plt.legend(
    bbox_to_anchor=(1.02,1),
    loc="upper left",
    borderaxespad=0
)

plt.tight_layout()

plt.savefig(
    "instruction_refinement_category_improvement.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ===============================
# Save category summary table
# ===============================

output_records = []

for category, curves in category_curves.items():

    curves = np.array(curves)

    final_values = curves[:, -1]

    record = {
        "category": category,
        "mean_mAP_improvement": np.mean(final_values),
        "num_datasets": len(curves)
    }

    output_records.append(record)

print("\nCategory Summary:")
for r in output_records:
    print(r)