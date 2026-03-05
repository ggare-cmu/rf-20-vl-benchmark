import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import os

base_dir = "results/final_consolidated_results/rf-20-vl-benchmark/results/eccv26/rf100vl_IPT/Qwen3-VL-30B-A3B-Instruct/rf20_IPT_singleclass_rankScore/iterative_prompt_refinement"

METRIC_KEY = "mAP_50_95"
num_iters = 11

dataset_dirs = sorted(glob.glob(os.path.join(base_dir, "*")))

plt.figure(figsize=(8,6))

for dataset_dir in dataset_dirs:

    dataset_name = os.path.basename(dataset_dir)

    files = glob.glob(
        os.path.join(dataset_dir, f"instruction_refinements_log_{dataset_name}_cls_*.json")
    )

    if len(files) == 0:
        continue

    all_results = []

    for file in files:

        cls_name = file.split('cls_')[-1].replace('.json', '')

        with open(file, "r") as f:
            data = json.load(f)

        iter_values = []
        for i in range(num_iters):
            key = f"class_{cls_name}_iter_{i}"

            if key in data:
                iter_values.append(data[key][METRIC_KEY])
            else:
                iter_values.append(np.nan)

        all_results.append(iter_values)

    all_results = np.array(all_results)

    mean_map = np.nanmean(all_results, axis=0)
    std_map = np.nanstd(all_results, axis=0)

    # 🔹 Enforce "best so far" curve
    mean_map = np.maximum.accumulate(mean_map)

    iterations = np.arange(num_iters)

    plt.plot(iterations, mean_map, marker='o', label=dataset_name)

    plt.fill_between(
        iterations,
        np.maximum.accumulate(mean_map - std_map),
        np.maximum.accumulate(mean_map + std_map),
        alpha=0.15
    )

plt.xlabel("Iteration")
plt.ylabel(METRIC_KEY)
plt.title("Instruction Refinement Performance Across Datasets")
plt.xticks(np.arange(num_iters))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("instruction_refinement_performance.png")
plt.show()