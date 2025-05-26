get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

declare -a dirs=(
    "workdir/agibot"
    "workdir/interiornet"
    "workdir/carla"
    "workdir/sceneflow"
    "workdir/droid"
    "workdir/navsim"
    "workdir/scand"
    "workdir/waymo"
    "workdir/drone"
)

method=iclight_vidtome_slicedit_opt

for dir in "${dirs[@]}"; do
    for outdir in $dir/$method/*; do
        while true; do
            gpu_id=$(get_available_gpu)
            if [[ -n $gpu_id ]]; then
                echo "GPU $gpu_id is available. Start evaluating '$outdir'"
                CUDA_VISIBLE_DEVICES=$gpu_id python evaluation/eval_video.py --output_dir $outdir --eval_cost &
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
done
wait

for dir in "${dirs[@]}"; do
    for outdir in $dir/$method/*; do
        while true; do
            gpu_id=$(get_available_gpu)
            if [[ -n $gpu_id ]]; then
                echo "GPU $gpu_id is available. Start evaluating '$outdir'"
                CUDA_VISIBLE_DEVICES=$gpu_id vbench evaluate \
                    --dimension motion_smoothness \
                    --videos_path $outdir/output.mp4 \
                    --output_path $outdir/vbench \
                    --mode=custom_input
                break
            else
                echo "No GPU available at the moment. Retrying in 2 minute."
                sleep 60
            fi
        done
    done
done
wait

for dir in "${dirs[@]}"; do
    python evaluation/avg_metrics.py --output_dirs $dir/$method/* --save_path $dir/$method.txt --vbench
    echo "Average metrics saved to $dir.txt"
done

python evaluation/avg_datasets_metrics.py --output_dirs "${dirs[@]}" --txt_name $method.txt

