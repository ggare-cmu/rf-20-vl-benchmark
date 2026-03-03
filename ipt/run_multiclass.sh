#!/bin/bash
set -e  # Exit if any command fails


WORKDIR=$(pwd)


# MODEL_NAME="Qwen2.5-VL-7B-Instruct"
# MODEL_NAME="Qwen2.5-VL-72B-Instruct"
# MODEL_NAME="Qwen3-VL-8B-Instruct"
MODEL_NAME="Qwen3-VL-30B-A3B-Instruct"

DATASET_ROOT="./datasets/rf100-vl-fsod/"
METHOD="gepa" #gepa, mipro
REFLECTION_LM="qwen" #qwen, openai

RESULTSDIR="../dspy-baselines/results/eccv26/$METHOD/$MODEL_NAME/rf20_"$METHOD"_REF_LM_"$REFLECTION_LM"_singleclass_rankScore"

# CMD_TEMPLATE="python3 baseline/evaluate_qwen_local_custom_instruction.py --data_dir $DATASET_ROOT --model_name $MODEL_NAME --save_dir $RESULTSDIR --vllm --just_instructions --instruction_type $METHOD --instruction_path $RESULTSDIR/$METHOD"
CMD_TEMPLATE="CUDA_VISIBLE_DEVICES=0,1,2,3 python3 baseline/evaluate_qwen_local_custom_instruction.py --data_dir $DATASET_ROOT --model_name $MODEL_NAME --save_dir $RESULTSDIR --vllm --just_instructions --instruction_type $METHOD --instruction_path $RESULTSDIR/$METHOD"
# CMD_TEMPLATE="source /home/ubuntu/miniforge3/etc/profile.d/conda.sh && conda activate qwen-vllm-env && python ipt/1run_rescorer.py --model_name $MODEL_NAME --ipt_mode --vqa_rescore --apply_nms --nms_threshold 0.5 --num_ipt_iterations 10 --output_dir $RESULTSDIR --vqa_batch_size 1"
# CMD_TEMPLATE="python ipt/1run_rescorer.py --model_name Qwen3-VL-235B-A22B-Instruct-FP8 --ipt_mode --vqa_rescore --apply_nms --nms_threshold 0.5 --num_ipt_iterations 10 --output_dir results/rf100vl_IPT/Qwen3-VL-235B-A22B-Instruct-FP8/rf20_IPT_singleclass_vqaScore_withNMS --vqa_batch_size 1"


# DATASETS=(
# #  "aerial-airport"
# #  "dentalai"
# #  "flir-camera-objects"
# #  "gwhd2021"
# #  "recode-waste"
# #  "wildfire-smoke"
# #  "x-ray-id"
# #  "soda-bottles"
# #  "wb-prova"
# #  "actions"
#   "aquarium-combined"
# #  "the-dreidel-project"
# #  "orionproducts"
# #  "trail-camera"
# #  "new-defects-in-wood"
# #  "lacrosse-object-detection"
#   "defect-detection"
#   "all-elements"
#   "water-meter"
# #  "paper-parts"
# )

# Create logs directory

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$WORKDIR/$RESULTSDIR/logs_$TIMESTAMP"

mkdir -p "$LOG_DIR"
echo "🗂 Logs directory: $LOG_DIR"


next_job="$CMD_TEMPLATE"
echo "Executing command: $next_job"

LOG_FILE="$LOG_DIR/multiclass.log"
echo "Logging to $LOG_FILE"

# Run with live streaming + logging
# 'stdbuf -oL' ensures line-buffered output for real-time viewing
stdbuf -oL -eL bash -c "$next_job" 2>&1 | tee "$LOG_FILE"
# eval "$next_job" 2>&1 | tee "$LOG_FILE"

echo "-------------------------------------------"


echo "Finished: multiclass"
echo


echo "✅ All datasets processed successfully!"
