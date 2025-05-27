get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

base_config="configs/scand/iclight_scand_vidtome.yaml"

declare -a configs=(
    "configs/sceneflow/scenes/iclight_sceneflow_35mm_bk_right.yaml"
    # "configs/scand/scenes/A_Spot_EER_OsCafe_Tue_Nov_9_39.yaml"
)

for config in configs/sceneflow/scenes/*; do
# for config in "${configs[@]}"; do
    while true; do
        gpu_id=$(get_available_gpu)
        if [[ -n $gpu_id ]]; then
            echo "GPU $gpu_id is available. Start running '$config'"
            # CUDA_VISIBLE_DEVICES=$gpu_id python run.py --config $config --base_config $base_config &
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
wait