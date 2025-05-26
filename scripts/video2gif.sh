get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

declare -a dirs=(
    "workdir/agibot"
    "workdir/carla"
    "workdir/droid"
    "workdir/drone"
    "workdir/interiornet"
    "workdir/navsim"
    "workdir/scand"
    "workdir/sceneflow"
    "workdir/waymo"
)

for dir in "${dirs[@]}"; do
    while true; do
        gpu_id=$(get_available_gpu)
        if [[ -n $gpu_id ]]; then
            echo "GPU $gpu_id is available. Start processing '$dir'"
            CUDA_VISIBLE_DEVICES=$gpu_id python tools/video_downsample.py -i $dir &
            # Allow some time for the process to initialize and potentially use GPU memory
            sleep 60
            # CUDA_VISIBLE_DEVICES=$gpu_id python evaluation/update_clipt.py --output_dir $outdir
            break
        else
            echo "No GPU available at the moment. Retrying in 2 minute."
            sleep 60
        fi
    done
done
wait

