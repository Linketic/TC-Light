import os
import sys
import argparse
import datetime
import torch
from omegaconf import OmegaConf
from plugin.FastBlend.api import smooth_video

# args.input_dir = 'workdir/agibot/iclight_vidtome'
mode = "Fast"
window_size = 15
batch_size = 16
guide_weight = 10.0
patch_size = 5
num_iter = 2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", help="input video file")
    args = parser.parse_args()


input_files = sorted(os.listdir(args.input_dir))
if len(input_files) == 0:
    print("No input files found in the directory.")
    sys.exit(1)

output_dir = args.input_dir + f'_fastblend_iter{num_iter}'

for file in input_files:

    config = OmegaConf.load(os.path.join(args.input_dir, file, 'config.yaml'))

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_time = datetime.datetime.now()

    smooth_video(
        video_guide = os.path.join(args.input_dir, file, "output_gt.mp4"),
        video_guide_folder = None,
        video_style = os.path.join(args.input_dir, file, "output.mp4"),
        video_style_folder = None,
        mode = mode,
        window_size = window_size,
        batch_size = batch_size,
        tracking_window_size = 1,
        output_path = os.path.join(output_dir, file),
        fps = None,
        minimum_patch_size = patch_size,
        num_iter = num_iter,
        guide_weight = guide_weight,
        initialize = "identity"
    )

    end_time = datetime.datetime.now()
    max_memory_allocated = torch.cuda.max_memory_allocated() / (1024.0 ** 2)

    config.total_time += (end_time - start_time).total_seconds()
    config.max_memory_allocated = max(config.max_memory_allocated, max_memory_allocated)
    config.sec_per_frame = config.total_time / config.total_number_of_frames
    # save config
    with open(os.path.join(output_dir, file, 'config.yaml'), 'w') as f:
        OmegaConf.save(config, f)
    
    os.system(f"cp {os.path.join(args.input_dir, file, 'output_gt.mp4')} {os.path.join(output_dir, file)}")