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

port=6041
method=iclight_vidtome_opt

for dir in "${dirs[@]}"; do
    for outdir in $dir/$method/*; do
        while true; do
            gpu_id=$(get_available_gpu)
            if [[ -n $gpu_id ]]; then
                echo "GPU $gpu_id is available. Start evaluating '$outdir'"
                CUDA_VISIBLE_DEVICES=$gpu_id python evaluation/evaluate_vbench.py \
                    --dimension motion_smoothness \
                            overall_consistency \
                            temporal_flickering \
                            aesthetic_quality \
                            imaging_quality \
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

python evaluation/avg_datasets_metrics.py --output_dirs workdir/* --txt_name $method.txt

