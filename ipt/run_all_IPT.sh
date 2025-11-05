#!/bin/bash

SESSION_NAME="rf20_IPT_singleclass_vqaScore_withNMS"
GPU_MEMORY_THRESHOLD=6000    # MB free memory needed to start a job
CHECK_INTERVAL=20            # seconds between GPU memory checks
WORKDIR=$(pwd)
# WORKDIR="/scratch/ggare/Research/VLMattributeClassifier"

# --- Configuration for job command ---

# MODEL_NAME="Qwen2.5-VL-7B-Instruct" #Qwen2.5-VL-7B-Instruct, Qwen2.5-VL-72B-Instruct, Qwen3-VL-30B-A3B-Instruct, Qwen3-VL-235B-A22B-Instruct
# MODEL_NAME="Qwen2.5-VL-72B-Instruct"
MODEL_NAME="Qwen3-VL-30B-A3B-Instruct"
# MODEL_NAME="Qwen3-VL-235B-A22B-Instruct"


# CUDA_VISIBLE_DEVICES=0 python ipt/run_bench_singleclass_VQAscoring_webUI_IPT_fs.py --ipt_mode --vqa_rescore --apply_nms --nms_threshold 0.5 --num_ipt_iterations 10 --output_dir results/rf100vl_IPT_tmp/rf20_IPT_singleclass_vqaScore_withNMS --vqa_batch_size 1
RESULTSDIR="results/rf100vl_IPT/Qwen3-VL-30B-A3B-Instruct/rf20_IPT_singleclass_vqaScore_withNMS"
CMD_TEMPLATE="conda activate rf100vl-env; python ipt/run_bench_singleclass_IPT.py --model_name $MODEL_NAME --ipt_mode --vqa_rescore --apply_nms --nms_threshold 0.5 --num_ipt_iterations 10 --output_dir $RESULTSDIR --vqa_batch_size 1"


# CMD_TEMPLATE="rb; python ipt/run_bench_singleclass_IPT.py --model_name Qwen2.5-VL-7B-Instruct --ipt_mode --vqa_rescore --apply_nms --nms_threshold 0.5 --num_ipt_iterations 10 --output_dir $RESULTSDIR --vqa_batch_size 1"

# # CUDA_VISIBLE_DEVICES=0 /data3/shared/scripts/vnice/vnice.sh python code/rf100vl/qwen-2.5-vl-rf-fsod-master/run_bench_singleclass_VQAscoring_webUI_multimetrics_flashAtt2.py --eval --vqa_rescore --apply_nms --nms_threshold 0.5 --output_dir results/rf100vl_zeroshot/rf20_singleclass_codePrompt_vqaScore_nms0.5_perClsInstr_flashAtt2 --vqa_batch_size 1
# RESULTSDIR="results/rf100vl_zeroshot/rf20_singleclass_codePrompt_vqaScore_nms0.5_perClsInstr_fixDetectScale"
# CMD_TEMPLATE="rb; /data3/shared/scripts/vnice/vnice.sh python code/rf100vl/qwen-2.5-vl-rf-fsod-master/run_bench_singleclass_VQAscoring_webUI_multimetrics_fixDetectScale.py --eval --vqa_rescore --apply_nms --nms_threshold 0.5 --output_dir $RESULTSDIR --vqa_batch_size 1"


# --- Create log directory ---
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$WORKDIR/$RESULTSDIR/logs_$TIMESTAMP"
mkdir -p "$LOG_DIR"
echo "🗂 Logs directory: $LOG_DIR"

# --- Detect number of GPUs dynamically ---
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "🔍 Detected $NUM_GPUS GPUs on this node."

# --- Job list ---
COMMANDS=(
    "$CMD_TEMPLATE --dataset_path aquarium-combined"
    "$CMD_TEMPLATE --dataset_path the-dreidel-project"
    "$CMD_TEMPLATE --dataset_path paper-parts"
    "$CMD_TEMPLATE --dataset_path x-ray-id"
    "$CMD_TEMPLATE --dataset_path gwhd2021"
    "$CMD_TEMPLATE --dataset_path aerial-airport"
    "$CMD_TEMPLATE --dataset_path recode-waste"
    "$CMD_TEMPLATE --dataset_path wb-prova"
    "$CMD_TEMPLATE --dataset_path water-meter"
    "$CMD_TEMPLATE --dataset_path dentalai"
    "$CMD_TEMPLATE --dataset_path trail-camera"
    "$CMD_TEMPLATE --dataset_path defect-detection"
    "$CMD_TEMPLATE --dataset_path wildfire-smoke"
    "$CMD_TEMPLATE --dataset_path actions"
    "$CMD_TEMPLATE --dataset_path all-elements"
    "$CMD_TEMPLATE --dataset_path orionproducts"
    "$CMD_TEMPLATE --dataset_path new-defects-in-wood"
    "$CMD_TEMPLATE --dataset_path soda-bottles"
    "$CMD_TEMPLATE --dataset_path flir-camera-objects"
    "$CMD_TEMPLATE --dataset_path lacrosse-object-detection"
)

# --- Shared job queue ---
QUEUE_FILE="$LOG_DIR/job_queue.txt"
printf "%s\n" "${COMMANDS[@]}" > "$QUEUE_FILE"
echo "📋 Job queue created: $QUEUE_FILE"

SUMMARY_FILE="$LOG_DIR/job_summary.txt"
echo "🧾 Job Summary (started at $(date))" > "$SUMMARY_FILE"
echo "------------------------------------" >> "$SUMMARY_FILE"

# --- Worker script ---
WORKER_SCRIPT="$LOG_DIR/worker.sh"
cat <<'EOF' > "$WORKER_SCRIPT"
#!/bin/bash
GPU_ID=$1
QUEUE_FILE=$2
CHECK_INTERVAL=$3
WORKDIR=$4
GPU_MEMORY_THRESHOLD=$5
LOG_DIR=$6
SUMMARY_FILE=$7

export CUDA_VISIBLE_DEVICES=$GPU_ID
cd "$WORKDIR" || exit 1

LOGFILE="$LOG_DIR/gpu_${GPU_ID}.log"
echo "📘 [$(date '+%Y-%m-%d %H:%M:%S')] Starting GPU worker $GPU_ID" | tee -a "$LOGFILE"

# --- Robust GPU free memory check ---
get_gpu_free_memory() {
    GPU_LINE=$((GPU_ID+1))
    # Capture numeric values only, strip non-numeric chars
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sed -n "${GPU_LINE}p" | tr -dc '0-9')
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n "${GPU_LINE}p" | tr -dc '0-9')

    # Handle missing or malformed output safely
    if [ -z "$total" ] || [ -z "$used" ]; then
        echo 0
        return
    fi

    # Ensure arithmetic is integer-safe
    free=$(( total - used ))
    if [ "$free" -lt 0 ]; then
        free=0
    fi
    echo "$free"
}

while true; do
    # Wait until GPU has enough free memory
    while true; do
        free_mem=$(get_gpu_free_memory)

        # Ensure numeric and non-empty
        if ! [[ "$free_mem" =~ ^[0-9]+$ ]]; then
            echo "⚠️ GPU $GPU_ID: Could not parse free memory value ('$free_mem'), retrying..." | tee -a "$LOGFILE"
            sleep "$CHECK_INTERVAL"
            continue
        fi

        if [ "$free_mem" -ge "$GPU_MEMORY_THRESHOLD" ]; then
            break
        fi

        echo "🕓 [$(date '+%H:%M:%S')] GPU $GPU_ID low free memory (${free_mem}MB < ${GPU_MEMORY_THRESHOLD}MB) — waiting..." | tee -a "$LOGFILE"
        sleep "$CHECK_INTERVAL"
    done

    # --- Lock queue and pick next job ---
    {
        flock -x 200
        next_job=$(head -n 1 "$QUEUE_FILE")
        if [ -n "$next_job" ]; then
            tail -n +2 "$QUEUE_FILE" > "$QUEUE_FILE.tmp" && mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"
        fi
    } 200>"$QUEUE_FILE.lock"

    if [ -z "$next_job" ]; then
        echo "✅ [$(date '+%H:%M:%S')] GPU $GPU_ID: No more jobs left. Exiting." | tee -a "$LOGFILE"
        break
    fi

    echo "🚀 [$(date '+%H:%M:%S')] GPU $GPU_ID starting job: $next_job" | tee -a "$LOGFILE"
    start_time=$(date +%s)

    # --- Run job with live output ---
    eval "$next_job" 2>&1 | tee -a "$LOGFILE"
    job_exit=$?

    end_time=$(date +%s)
    duration=$((end_time - start_time))
    job_name=$(echo "$next_job" | awk -F'--dataset_path ' '{print $2}' | awk '{print $1}')

    if [ $job_exit -eq 0 ]; then
        echo "🎯 [$(date '+%H:%M:%S')] GPU $GPU_ID finished job successfully in ${duration}s" | tee -a "$LOGFILE"
        echo "✅ SUCCESS | GPU:$GPU_ID | Job:$job_name | Duration:${duration}s | $(date)" >> "$SUMMARY_FILE"
    else
        echo "❌ [$(date '+%H:%M:%S')] GPU $GPU_ID job FAILED (exit code $job_exit) after ${duration}s" | tee -a "$LOGFILE"
        echo "❌ FAILURE | GPU:$GPU_ID | Job:$job_name | Duration:${duration}s | $(date)" >> "$SUMMARY_FILE"
    fi

    sleep "$CHECK_INTERVAL"
done

echo "🎉 [$(date '+%Y-%m-%d %H:%M:%S')] GPU $GPU_ID done with all jobs!" | tee -a "$LOGFILE"
EOF

chmod +x "$WORKER_SCRIPT"

# --- Create tmux session and panes ---
tmux new-session -d -s "$SESSION_NAME" "echo '🧠 Starting dynamic GPU job manager...'; bash"
sleep 1

for ((gpu=1; gpu<NUM_GPUS; gpu++)); do
    tmux split-window -t "$SESSION_NAME":0 -h
    tmux select-layout -t "$SESSION_NAME":0 tiled >/dev/null
    sleep 0.2
done

# --- Launch workers ---
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    tmux send-keys -t "$SESSION_NAME":0.$gpu \
        "bash $WORKER_SCRIPT $gpu $QUEUE_FILE $CHECK_INTERVAL $WORKDIR $GPU_MEMORY_THRESHOLD $LOG_DIR $SUMMARY_FILE" C-m
done

tmux select-layout -t "$SESSION_NAME":0 tiled

echo "🎬 All dynamic GPU workers started in tmux session: $SESSION_NAME"
echo "👉 Attach: tmux attach -t $SESSION_NAME"
echo "👉 Detach: Ctrl+b then d"
echo "👉 Logs: $LOG_DIR"
echo "👉 Summary: $SUMMARY_FILE"

# --- Final summary watcher ---
(
    echo ""
    echo "🧾 Monitoring job completion..."
    while pgrep -f "$WORKER_SCRIPT" >/dev/null; do
        sleep 30
    done
    echo "✅ All workers have finished!"
    echo ""
    echo "📜 Final Job Summary:"
    echo "------------------------------------"
    cat "$SUMMARY_FILE"
) &
