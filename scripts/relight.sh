get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

base_config="configs/tclight_default.yaml"

declare -a configs=(
    "configs/examples/tclight_droid.yaml"
    "configs/examples/tclight_navsim.yaml"
    "configs/examples/tclight_scand.yaml"
)

for config in "${configs[@]}"; do
    while true; do
        gpu_id=$(get_available_gpu)
        if [[ -n $gpu_id ]]; then
            echo "GPU $gpu_id is available. Start running '$config'"
            
            # if you want to appoint base_config, add --base_config $base_config
            CUDA_VISIBLE_DEVICES=$gpu_id python run.py --config $config &
            
            # Allow some time for the process to initialize and potentially use GPU memory
            sleep 60
            break
        else
            echo "No GPU available at the moment. Retrying in 2 minute."
            sleep 60
        fi
    done
done
wait