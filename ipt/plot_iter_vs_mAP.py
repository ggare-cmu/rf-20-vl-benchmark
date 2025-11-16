import matplotlib.pyplot as plt
import re

import seaborn as sns
import pandas as pd

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

# -----------------------------------------------------------
# 2. Convert dict → tidy DataFrame for seaborn
# -----------------------------------------------------------
pattern = r"class_(\w+)_iter_(\d+)"
rows = []

for key, val in results.items():
    m = re.match(pattern, key)
    if m:
        cls = m.group(1)
        iter_idx = int(m.group(2))
        rows.append({
            "Class": cls,
            "Iteration": iter_idx,
            "mAP_50_95": val["mAP_50_95"],
        })

df = pd.DataFrame(rows)

# Sort classes consistently (Adult, Juvenile, Piglet)
order = ["Adult", "Juvenile", "Piglet"]
df["Class"] = pd.Categorical(df["Class"], categories=order, ordered=True)

# -----------------------------------------------------------
# 3. Seaborn styling for CVPR paper
# -----------------------------------------------------------
sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
sns.set_palette("colorblind")

plt.figure(figsize=(8, 5))

sns.lineplot(
    data=df,
    x="Iteration",
    y="mAP_50_95",
    hue="Class",
    marker="o",
    linewidth=2.5,
    markersize=9,
)

# -----------------------------------------------------------
# 4. Styling tweaks for publication quality
# -----------------------------------------------------------
plt.xlabel("Iteration", fontsize=16)
plt.ylabel("mAP [50–95]", fontsize=16)
plt.title("Iterative Instruction Optimization: mAP vs Iteration", fontsize=17)

plt.grid(True, linewidth=0.8, alpha=0.4)
plt.legend(title="Class", fontsize=13, title_fontsize=13)
plt.tight_layout()

plt.savefig("iter_vs_mAP_classes_seaborn.png")
# plt.show()
