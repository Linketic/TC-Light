get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

declare -a datasets=(
    "agibot"
    "carla"
    "droid"
    "drone"
    "interiornet"
    "navsim"
    "scand"
    "sceneflow"
    "waymo"
)


for dataset in "${datasets[@]}"; do
    base_config="plugin/VidToMe/configs/$dataset/iclight_${dataset}_vidtome_opt_uvt2nd_hardmsk_flow.yaml"
    for config in plugin/VidToMe/configs/$dataset/scenes/*; do
        while true; do
            gpu_id=$(get_available_gpu)
            if [[ -n $gpu_id ]]; then
                echo "GPU $gpu_id is available. Start running '$config'"
                CUDA_VISIBLE_DEVICES=$gpu_id python run.py --config $config --base_config $base_config &
                # CUDA_VISIBLE_DEVICES=$gpu_id python run.py --config $config & 
                # Allow some time for the process to initialize and potentially use GPU memory
                sleep 60
                break
            elsed
                echo "No GPU available at the moment. Retrying in 2 minute."
                sleep 60
            fi
        done
    done
done
wait

