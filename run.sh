get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

declare -a configs=(
    # "plugin/VidToMe/configs/iclight_sceneflow_opt.yaml"
    "plugin/VidToMe/configs/iclight_sceneflow_opt_lmr5_gmr4.yaml"
    "plugin/VidToMe/configs/iclight_sceneflow_opt_lmr9_gmr8.yaml"
)

for config in "${configs[@]}"; do
    while true; do
        gpu_id=$(get_available_gpu)
        if [[ -n $gpu_id ]]; then
            echo "GPU $gpu_id is available. Start running '$config'"
            CUDA_VISIBLE_DEVICES=$gpu_id python run.py --config $config &
            # Allow some time for the process to initialize and potentially use GPU memory
            sleep 120
            break
        else
            echo "No GPU available at the moment. Retrying in 2 minute."
            sleep 120
        fi
    done
done
