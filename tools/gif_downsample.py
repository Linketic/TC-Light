import os
from PIL import Image
from PIL import ImageSequence
from tqdm import tqdm

input_dirs = [
    # "workdir/sceneflow/15mm_bk_left_lmr_0.9_gmr_0.8_vox_0.02_opt",
    # "workdir/sceneflow/15mm_fw_right_tokyo_lmr_0.9_gmr_0.8_vox_0.02_opt",
    # "workdir/sceneflow/35mm_bk_right_natural_lmr_0.9_gmr_0.8_vox_0.02_opt",
    # "workdir/sceneflow/35mm_fw_left_winter_lmr_0.9_gmr_0.8_vox_0.02_opt",
    # "workdir/agirobot_digital/agibot_digitaltwin_0_lmr_0.9_gmr_0.8_vox_None_opt",
    # "workdir/agirobot_digital/agirobot_digitaltwin_1_lmr_0.9_gmr_0.8_vox_None_opt",
    # "workdir/agirobot_digital/agirobot_digitaltwin_2_warm_lmr_0.9_gmr_0.8_vox_None_opt",
    # "workdir/agirobot_digital/agirobot_digitaltwin_3_studio_lmr_0.9_gmr_0.8_vox_None_opt",
    # "workdir/agirobot_digital/agirobot_digitaltwin_4_shadow_lmr_0.9_gmr_0.8_vox_None_opt",
    # "workdir/agirobot_digital/agirobot_digitaltwin_5_natural_lmr_0.9_gmr_0.8_vox_None_opt",
    # "workdir/agirobot_digital/agirobot_digitaltwin_6_sunshine_lmr_0.9_gmr_0.8_vox_None_opt",
    # "workdir/agirobot_digital/agibot_digitaltwin_7_cinematic_lmr_0.9_gmr_0.8_vox_None_opt",
    "workdir/agirobot_digital/agirobot_digitaltwin_5_natural_lmr_0.9_gmr_0.8_vox_None/exposure",
    "workdir/agirobot_digital/agirobot_digitaltwin_5_natural_lmr_0.9_gmr_0.8_vox_None/rife_smoothed_2"
]

out_dir = "workdir/demos"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

downsample_factor = 3

for input_dir in tqdm(input_dirs, desc="Downsampling GIFs"):

    file_name = input_dir.split("/")[-1]

    im = Image.open(f'{input_dir}/output.gif')
    resize_frames= [frame.resize((frame.width // downsample_factor, frame.height // downsample_factor)) for frame in ImageSequence.Iterator(im)]
    resize_frames[0].save(f"{out_dir}/{file_name}_ds{downsample_factor}.gif", save_all=True, append_images=resize_frames[1:])

    im_gt = Image.open(f'{input_dir}/output_gt.gif')
    resize_frames_gt= [frame.resize((frame.width // downsample_factor, frame.height // downsample_factor)) for frame in ImageSequence.Iterator(im_gt)]
    resize_frames_gt[0].save(f"{out_dir}/{file_name}_gt_ds{downsample_factor}.gif", save_all=True, append_images=resize_frames_gt[1:])

print("Done")