#!/bin/bash
set -e  # Exit if any command fails


WORKDIR=$(pwd)


# MODEL_NAME="Qwen2.5-VL-7B-Instruct" #Qwen2.5-VL-7B-Instruct, Qwen2.5-VL-72B-Instruct, Qwen3-VL-30B-A3B-Instruct, Qwen3-VL-235B-A22B-Instruct
MODEL_NAME="Qwen2.5-VL-72B-Instruct"
# MODEL_NAME="Qwen3-VL-8B-Instruct"
# MODEL_NAME="Qwen3-VL-30B-A3B-Instruct"
# MODEL_NAME="Qwen3-VL-235B-A22B-Instruct"
# MODEL_NAME="Qwen3-VL-235B-A22B-Instruct-FP8"


# CMD_TEMPLATE="python ipt/run_bench_singleclass_IPT.py --model_name Qwen3-VL-235B-A22B-Instruct-FP8 --ipt_mode --vqa_rescore --apply_nms --nms_threshold 0.5 --num_ipt_iterations 10 --output_dir results/rf100vl_IPT/Qwen3-VL-235B-A22B-Instruct-FP8/rf20_IPT_singleclass_vqaScore_withNMS --vqa_batch_size 1"
RESULTSDIR="results/rf100vl_IPT/$MODEL_NAME/rf20_IPT_singleclass_vqaScore_withNMS"
CMD_TEMPLATE="source /home/ubuntu/miniforge3/etc/profile.d/conda.sh && conda activate qwen-vllm-env && python ipt/run_bench_singleclass_IPT.py --model_name $MODEL_NAME --ipt_mode --vqa_rescore --apply_nms --nms_threshold 0.5 --num_ipt_iterations 10 --output_dir $RESULTSDIR --vqa_batch_size 1"


DATASETS=(
  "aquarium-combined"
  "the-dreidel-project"
  "paper-parts"
  "x-ray-id"
  "gwhd2021"
  "aerial-airport"
  "recode-waste"
  "wb-prova"
  "water-meter"
  "dentalai"
  "trail-camera"
  "defect-detection"
  "wildfire-smoke"
  "actions"
  "all-elements"
  "orionproducts"
  "new-defects-in-wood"
  "soda-bottles"
  "flir-camera-objects"
  "lacrosse-object-detection"
)

# Create logs directory
mkdir -p logs

for dataset in "${DATASETS[@]}"; do
    echo "==========================================="
    echo "Running dataset: $dataset"
    echo "==========================================="
    CMD="$CMD_TEMPLATE --dataset_path $dataset"
    
    LOG_FILE="logs/${dataset}.log"
    echo "Logging to $LOG_FILE"
    
    # Run command and log both stdout and stderr
    eval "$CMD" > "$LOG_FILE" 2>&1
    
    echo "Finished: $dataset"
    echo
done

echo "✅ All datasets processed successfully!"
