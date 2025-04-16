get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

base_config="plugin/VidToMe/configs/agibot/iclight_agibot.yaml"

declare -a configs=(
    "plugin/VidToMe/configs/agibot/scenes/digitaltwin_6.yaml"
    "plugin/VidToMe/configs/agibot/scenes/digitaltwin_1.yaml"
)

for config in plugin/VidToMe/configs/agibot/scenes/*; do
# for config in "${configs[@]}"; do
    while true; do
        gpu_id=$(get_available_gpu)
        if [[ -n $gpu_id ]]; then
            # echo "GPU $gpu_id is available. Start running '$config'"
            CUDA_VISIBLE_DEVICES=$gpu_id python run.py --config $config --base_config $base_config &
            # CUDA_VISIBLE_DEVICES=$gpu_id python run.py --config $config & 
            # Allow some time for the process to initialize and potentially use GPU memory
            sleep 120
            break
        else
            echo "No GPU available at the moment. Retrying in 2 minute."
            sleep 120
        fi
    done
done