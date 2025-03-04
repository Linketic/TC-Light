get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

declare -a outdirs=(
    # "workdir/sceneflow/15mm_bk_left_lmr_0.01_gmr_0.01_vox_None"
    "workdir/sceneflow/15mm_bk_left_lmr_0.9_gmr_0.8_vox_0.02_opt"
    # "workdir/sceneflow/15mm_bk_left_lmr_0.9_gmr_0.8_vox_None"
    # "workdir/sceneflow/15mm_fw_right_tokyo_lmr_0.01_gmr_0.01_vox_None"
    "workdir/sceneflow/15mm_fw_right_tokyo_lmr_0.9_gmr_0.8_vox_0.02_opt"
    # "workdir/sceneflow/15mm_fw_right_tokyo_lmr_0.9_gmr_0.8_vox_None"
    # "workdir/sceneflow/35mm_bk_right_natural_lmr_0.01_gmr_0.01_vox_None"
    "workdir/sceneflow/35mm_bk_right_natural_lmr_0.9_gmr_0.8_vox_0.02_opt"
    # "workdir/sceneflow/35mm_bk_right_natural_lmr_0.9_gmr_0.8_vox_None"
    # "workdir/sceneflow/35mm_fw_left_winter_lmr_0.01_gmr_0.01_vox_None"
    "workdir/sceneflow/35mm_fw_left_winter_lmr_0.9_gmr_0.8_vox_0.02_opt"
    # "workdir/sceneflow/35mm_fw_left_winter_lmr_0.9_gmr_0.8_vox_None"
)

for outdir in "${outdirs[@]}"; do
    while true; do
        gpu_id=$(get_available_gpu)
        if [[ -n $gpu_id ]]; then
            echo "GPU $gpu_id is available. Start transforming '$outdir'"
            CUDA_VISIBLE_DEVICES=$gpu_id python tools/img2video.py \
                                                -i $outdir/frames \
                                                -o $outdir/output.gif \
                                                -m 960 -f 15 &
            # Allow some time for the process to initialize and potentially use GPU memory
            sleep 60
            break
        else
            echo "No GPU available at the moment. Retrying in 2 minute."
            sleep 60
        fi
    done
done
