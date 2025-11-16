#!/bin/bash
set -e  # Exit if any command fails


WORKDIR=$(pwd)


# MODEL_NAME="Qwen2.5-VL-7B-Instruct" #Qwen2.5-VL-7B-Instruct, Qwen2.5-VL-72B-Instruct, Qwen3-VL-30B-A3B-Instruct, Qwen3-VL-235B-A22B-Instruct
# MODEL_NAME="Qwen2.5-VL-72B-Instruct"
# MODEL_NAME="Qwen3-VL-8B-Instruct"
MODEL_NAME="Qwen3-VL-30B-A3B-Instruct"
# MODEL_NAME="Qwen3-VL-235B-A22B-Instruct"
# MODEL_NAME="Qwen3-VL-235B-A22B-Instruct-FP8"


# CMD_TEMPLATE="python ipt/run_bench_singleclass_IPT.py --model_name Qwen3-VL-235B-A22B-Instruct-FP8 --ipt_mode --vqa_rescore --apply_nms --nms_threshold 0.5 --num_ipt_iterations 10 --output_dir results/rf100vl_IPT/Qwen3-VL-235B-A22B-Instruct-FP8/rf20_IPT_singleclass_vqaScore_withNMS --vqa_batch_size 1"
DATA_INSTR_PATH="./data_instr/default/README.dataset"
# DATA_INSTR_PATH="./results/completed_exp/rf100vl_IPT/Qwen3-VL-30B-A3B-Instruct/iterative_prompt_refinement/all_refined_class_instructions"
# DATA_INSTR_PATH="./results/completed_exp/rf100vl_IPT/Qwen2.5-VL-7B-Instruct/iterative_prompt_refinement/all_refined_class_instructions"
# RESULTSDIR="results/rf100vl_IPT_ablations/$MODEL_NAME/rf20_IPT_singleclass_vqaScore_withNMS_withQwen2-7BDataInstr"
RESULTSDIR="results/rf100vl_IPT_ablations/$MODEL_NAME/rf20_IPT_singleclass_vqaScore_withNMS_defaultDataInstr"
CMD_TEMPLATE="python ipt/run_bench_singleclass_evaluator.py --model_name $MODEL_NAME --vqa_rescore --apply_nms --nms_threshold 0.5 --data_instr_path $DATA_INSTR_PATH --output_dir $RESULTSDIR --vqa_batch_size 1"
# CMD_TEMPLATE="python ipt/run_bench_singleclass_evaluator.py --model_name $MODEL_NAME --data_instr_path $DATA_INSTR_PATH --output_dir $RESULTSDIR --vqa_batch_size 1"
# CMD_TEMPLATE="source /home/ubuntu/miniforge3/etc/profile.d/conda.sh && conda activate qwen-vllm-env && python ipt/run_bench_singleclass_IPT.py --model_name $MODEL_NAME --ipt_mode --vqa_rescore --apply_nms --nms_threshold 0.5 --num_ipt_iterations 10 --output_dir $RESULTSDIR --vqa_batch_size 1"


DATASETS=(
  "aerial-airport"
  "dentalai"
  "flir-camera-objects"
  "gwhd2021"
  "recode-waste"
  "wildfire-smoke"
  "x-ray-id"
  "soda-bottles"
  "wb-prova"
  "actions"
  "aquarium-combined"
  "the-dreidel-project"
  "orionproducts"
  "trail-camera"
  "new-defects-in-wood"
  "lacrosse-object-detection"
  "defect-detection"
  "all-elements"
  "water-meter"
  "paper-parts"
)

# Create logs directory

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$WORKDIR/$RESULTSDIR/logs_$TIMESTAMP"

mkdir -p "$LOG_DIR"
echo "🗂 Logs directory: $LOG_DIR"


for dataset in "${DATASETS[@]}"; do
    echo "==========================================="
    echo "Running dataset: $dataset"
    echo "==========================================="
    
    next_job="$CMD_TEMPLATE --dataset_path $dataset"
    echo "Executing command: $next_job"

    LOG_FILE="$LOG_DIR/${dataset}.log"
    echo "Logging to $LOG_FILE"
    
    # Run with live streaming + logging
    # 'stdbuf -oL' ensures line-buffered output for real-time viewing
    stdbuf -oL -eL bash -c "$next_job" 2>&1 | tee "$LOG_FILE"
    # eval "$next_job" 2>&1 | tee "$LOG_FILE"

    echo "-------------------------------------------"
    
    
    echo "Finished: $dataset"
    echo
done

echo "✅ All datasets processed successfully!"
