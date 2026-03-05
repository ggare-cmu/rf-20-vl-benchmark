import matplotlib.pyplot as plt
import re

import json

# -----------------------------------------------------------
# 1. INSERT your complete results dictionary here:
#    Must contain entries like:
#    "class_Adult_iter_0", "class_Juvenile_iter_3", "class_Piglet_iter_7"
# -----------------------------------------------------------
# results = {
#     # --- Piglet entries (example) ---
#     "class_Piglet_iter_0": {"mAP_50_95": 0.61518},
#     "class_Piglet_iter_1": {"mAP_50_95": 0.57977},
#     "class_Piglet_iter_2": {"mAP_50_95": 0.63079},
#     # ... add all piglet entries ...
    
#     # --- Juvenile entries ---
#     # "class_Juvenile_iter_0": {...},
#     # "class_Juvenile_iter_1": {...},
    
#     # --- Adult entries ---
#     # "class_Adult_iter_0": {...},
#     # "class_Adult_iter_1": {...},
# }

data_path = "results/lambda_results/132.145.195.234/results/rf100vl_IPT/Qwen3-VL-30B-A3B-Instruct/rf20_IPT_singleclass_vqaScore_withNMS/iterative_prompt_refinement/wb-prova/instruction_refinements_log_wb-prova_cls"

classes = ["Adult", "Juvenile", "Piglet"]

# Load results from files
results = {}
for cls in classes:
    file_path = f"{data_path}_{cls}.json"
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            # for iter_idx, metrics in data.items():
            #     mAP_value = metrics.get("mAP_50_95", 0)
            #     results[f"class_{cls}_iter_{iter_idx}"] = {"mAP_50_95": mAP_value}
            results.update(data)

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

# -----------------------------------------------------------
# 2. Parse keys of the form: class_<ClassName>_iter_<N>
# -----------------------------------------------------------
pattern = r"class_(\w+)_iter_(\d+)"
class_data = {}   # { "Piglet": {iters: [], mAP: []}, ... }

for key, val in results.items():
    match = re.match(pattern, key)
    if match:
        cls = match.group(1)
        iter_idx = int(match.group(2))
        mAP = val["mAP_50_95"]

        if cls not in class_data:
            class_data[cls] = {"iters": [], "mAP": []}

        class_data[cls]["iters"].append(iter_idx + 1)
        class_data[cls]["mAP"].append(mAP)

# -----------------------------------------------------------
# 3. Plot for each class
# -----------------------------------------------------------
plt.figure(figsize=(8,6))

for cls, data in class_data.items():
    # Sort by iteration index
    iters, mAPs = zip(*sorted(zip(data["iters"], data["mAP"])))

    plt.plot(
        iters,
        mAPs,
        marker='o',
        linewidth=2,
        label=cls
    )

plt.title("Iteration vs mAP[50–95] for Adult, Juvenile, Piglet")
plt.xlabel("Iteration")
plt.ylabel("mAP[50–95]")
plt.grid(True, linestyle="--", alpha=0.5)
plt.xticks(sorted(set(sum([data["iters"] for data in class_data.values()], []))))
plt.legend()
plt.tight_layout()
plt.savefig("iter_vs_mAP_classes1.png")
plt.show()
