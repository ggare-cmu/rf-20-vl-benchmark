import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import os

base_dir = "results/final_consolidated_results/rf-20-vl-benchmark/results/eccv26/rf100vl_IPT/Qwen3-VL-30B-A3B-Instruct/rf20_IPT_singleclass_rankScore/iterative_prompt_refinement"

METRIC_KEY = "mAP_50_95"
num_iters = 11

dataset_dirs = sorted(glob.glob(os.path.join(base_dir, "*")))

# large figure so legend fits
plt.figure(figsize=(12,7))

# color palette for up to 20 datasets
colors = plt.cm.tab20(np.linspace(0,1,len(dataset_dirs)))

for idx, dataset_dir in enumerate(dataset_dirs):

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

        # best-so-far per class
        iter_values = np.maximum.accumulate(iter_values)

        all_results.append(iter_values)

    all_results = np.array(all_results)

    mean_map = np.nanmean(all_results, axis=0)
    std_map = np.nanstd(all_results, axis=0)

    # convert to %
    mean_map *= 100
    std_map *= 100

    iterations = np.arange(num_iters)

    plt.plot(
        iterations,
        mean_map,
        marker='o',
        linewidth=2,
        color=colors[idx],
        label=dataset_name
    )

    # plt.fill_between(
    #     iterations,
    #     mean_map - std_map,
    #     mean_map + std_map,
    #     alpha=0.12,
    #     color=colors[idx]
    # )

# fixed y-axis
# plt.ylim(0,100)
plt.ylim(0,60)

plt.xlabel("Iteration", fontsize=13)
plt.ylabel("mAP@50:95 (%)", fontsize=13)
plt.title("Instruction Refinement Performance Across 20 Datasets", fontsize=14)

plt.xticks(np.arange(num_iters))
plt.grid(True, linestyle="--", alpha=0.4)

# legend outside plot
plt.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    borderaxespad=0,
    fontsize=9
)

plt.tight_layout()

plt.savefig(
    "instruction_refinement_performance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()