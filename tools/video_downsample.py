import os
import io
import cv2
import argparse
from PIL import Image
from PIL import ImageSequence
from tqdm import tqdm

def estimate_gif_size(frames, fps):
    buffer = io.BytesIO()
    frames[0].save(
        buffer,
        format='GIF',
        save_all=True,
        append_images=frames[1:],
        duration=1000 / fps,
        loop=0,
        optimize=True
    )
    return buffer.tell()

def frames_resize(frames, factor):
    return [frame.resize(
        (frame.width // factor, frame.height // factor),
        Image.LANCZOS
    ) for frame in frames]

def find_optimal_downsample_factor(frames, fps, min_size, max_size, max_factor=10):
    factor = 1
    while factor <= max_factor:
        resized = frames_resize(frames, factor)
        size = estimate_gif_size(resized, fps)
        if min_size <= size <= max_size:
            return factor, resized
        factor += 1
    return None, None

# input_dirs = [
#     "workdir/navsim/iclight_vidtome_opt/navsim_16_00634_01421_lmr_0.6_gmr_0.5_alpha_t_0.0_opt",
#     "workdir/navsim/iclight_vidtome_opt/navsim_17_01352_0190_lmr_0.6_gmr_0.5_alpha_t_0.0_opt",
#     "workdir/navsim/iclight_vidtome_opt/navsim_26_03873_04225_lmr_0.6_gmr_0.5_alpha_t_0.0_opt",
#     "workdir/navsim/iclight_vidtome_opt/navsim_38_00305_00597_lmr_0.6_gmr_0.5_alpha_t_0.0_opt",
#     "workdir/navsim/iclight_vidtome_opt/navsim_50_01085_01463_lmr_0.6_gmr_0.5_alpha_t_0.0_opt"
# ]

# methods = ["cosmos_t1", "iclight", "iclight_vidtome", "iclight_vidtome_opt", "iclight_vidtome_slicedit_opt", "slicedit"]

methods = ["slicedit_org_size"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", help="input scene folder", default="workdir/carla")
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = input_folder.replace("workdir", "workdir/demos")
    prompts = {}

    for method in methods:
        input_dirs = os.path.join(input_folder, method)
        out_dir = os.path.join(output_folder, method)
        gt_dir = os.path.join(output_folder, "gt")
        input_dirs = [os.path.join(input_dirs, d) for d in sorted(os.listdir(input_dirs)) if os.path.isdir(os.path.join(input_dirs, d))]

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if not os.path.exists(gt_dir):
            os.makedirs(gt_dir)

        for input_dir in tqdm(input_dirs, desc=f"Downsampling {method} videos..."):
            
            file_name = input_dir.split("/")[-1]

            # read the file with cv2 and transform each frame to PIL
            cap = cv2.VideoCapture(f'{input_dir}/output.mp4')
            cap_gt = cv2.VideoCapture(f'{input_dir}/output_gt.mp4')
            fps = cap.get(cv2.CAP_PROP_FPS)
            im = []
            im_gt = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                im.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            cap.release()

            # Define size limits in bytes
            min_size_bytes = 10 * 1024 * 1024  # 10MB
            max_size_bytes = 20 * 1024 * 1024  # 20MB

            # Find optimal downsample factor
            # optimal_factor, resized_frames = find_optimal_downsample_factor(
            #     im, fps, min_size_bytes, max_size_bytes
            # )
            optimal_factor = 3
            resized_frames = frames_resize(im, optimal_factor)

            if optimal_factor is None:
                print(f"Could not find a suitable downsample factor for {file_name}.")
                continue

            if method == "cosmos_t1":
                while True:
                    ret, frame = cap_gt.read()
                    if not ret:
                        break
                    im_gt.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                cap_gt.release()
                # optimal_factor, resize_frames_gt = find_optimal_downsample_factor(
                #     im_gt, fps, min_size_bytes, max_size_bytes
                # )
                optimal_factor = 3
                resize_frames_gt = frames_resize(im_gt, optimal_factor)

            file_name = file_name.replace(":", "_")
            resized_frames[0].save(f"{out_dir}/{file_name}_ds{optimal_factor}.gif", save_all=True, append_images=resized_frames[1:], duration=1000/fps, loop=0)
            if method == "cosmos_t1":
                from omegaconf import OmegaConf
                config = OmegaConf.load(f'{input_dir}/config.yaml')
                prompts.update({list(config.generation.prompt.keys())[0]: list(config.generation.prompt.values())[0]})
                if optimal_factor is None:
                    print(f"Could not find a suitable downsample factor for {file_name}_gt.")
                    continue
                resize_frames_gt[0].save(f"{gt_dir}/{file_name}_gt_ds{optimal_factor}.gif", save_all=True, append_images=resize_frames_gt[1:], duration=1000/fps, loop=0)

        if method == "cosmos_t1":
            txt_save_path = os.path.join(output_folder, "prompts.txt")
            with open(txt_save_path, "w") as f:
                for key, value in prompts.items():
                    f.write(f"{key}: {value}\n")
            
            print(f"Prompts saved to {txt_save_path}")

    print("Done")