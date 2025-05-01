get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

declare -a dirs=(
    "workdir/agibot/iclight"
    "workdir/agibot/iclight_vidtome"
    "workdir/agibot/iclight_vidtome_fastblend"
    "workdir/agibot/iclight_vidtome_opt"
    "workdir/agibot/iclight_vidtome_slicedit_opt"
    "workdir/droid/iclight"
    "workdir/droid/iclight_vidtome"
    "workdir/droid/iclight_vidtome_fastblend"
    "workdir/droid/iclight_vidtome_opt"
    "workdir/droid/iclight_vidtome_slicedit_opt"
    "workdir/drone/iclight"
    "workdir/drone/iclight_vidtome"
    "workdir/drone/iclight_vidtome_fastblend"
    "workdir/drone/iclight_vidtome_opt"
    "workdir/drone/iclight_vidtome_slicedit_opt"
    "workdir/interiornet/iclight"
    "workdir/interiornet/iclight_vidtome"
    "workdir/interiornet/iclight_vidtome_fastblend"
    "workdir/interiornet/iclight_vidtome_opt"
    "workdir/interiornet/iclight_vidtome_slicedit_opt"
    "workdir/navsim/iclight"
    "workdir/navsim/iclight_vidtome"
    "workdir/navsim/iclight_vidtome_fastblend"
    "workdir/navsim/iclight_vidtome_opt"
    "workdir/navsim/iclight_vidtome_slicedit_opt"
    "workdir/scand/iclight"
    "workdir/scand/iclight_vidtome"
    "workdir/scand/iclight_vidtome_fastblend"
    "workdir/scand/iclight_vidtome_opt"
    "workdir/scand/iclight_vidtome_slicedit_opt"
    "workdir/waymo/iclight"
    "workdir/waymo/iclight_vidtome"
    "workdir/waymo/iclight_vidtome_fastblend"
    "workdir/waymo/iclight_vidtome_opt"
    "workdir/waymo/iclight_vidtome_slicedit_opt"
)

# dir=workdir/sceneflow/iclight_vidtome

for dir in "${dirs[@]}"; do
    for outdir in $dir/*; do
        while true; do
            gpu_id=$(get_available_gpu)
            if [[ -n $gpu_id ]]; then
                echo "GPU $gpu_id is available. Start evaluating '$outdir'"
                CUDA_VISIBLE_DEVICES=$gpu_id python evaluation/eval_video.py --output_dir $outdir --eval_cost &
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

    python evaluation/avg_metrics.py --output_dirs $dir/* --save_path $dir.txt
done

