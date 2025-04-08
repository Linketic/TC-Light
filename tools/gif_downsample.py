import os
from PIL import Image
from PIL import ImageSequence
from tqdm import tqdm

input_dirs = [
    "workdir/waymo/waymo_200_lmr_0.6_gmr_0.5_alpha_t_0.1_opt",
    "workdir/waymo/waymo_253_lmr_0.6_gmr_0.5_alpha_t_0.1_opt",
    "workdir/waymo/waymo_492_lmr_0.6_gmr_0.5_alpha_t_0.1_opt",
    "workdir/waymo/waymo_648_lmr_0.6_gmr_0.5_alpha_t_0.1_opt",
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