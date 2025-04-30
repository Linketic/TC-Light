get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

declare -a input_dirs=(
    "workdir/sceneflow/iclight_vidtome"
    "workdir/carla/iclight_vidtome"
    "workdir/droid/iclight_vidtome"
    "workdir/drone/iclight_vidtome"
    "workdir/interiornet/iclight_vidtome"
    "workdir/navsim/iclight_vidtome"
    "workdir/scand/iclight_vidtome"
    "workdir/sceneflow/iclight_vidtome"
    "workdir/waymo/iclight_vidtome"
)

for input_dir in "${input_dirs[@]}"; do
    while true; do
        gpu_id=$(get_available_gpu)
        if [[ -n $gpu_id ]]; then
            echo "GPU $gpu_id is available. Start running '$input_dir'"
            CUDA_VISIBLE_DEVICES=$gpu_id python run_fastblend.py -i $input_dir &
            # Allow some time for the process to initialize and potentially use GPU memory
            sleep 120
            break
        elsed
            echo "No GPU available at the moment. Retrying in 2 minute."
            sleep 120
        fi
    done
done
wait