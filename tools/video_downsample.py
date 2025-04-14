import os
import cv2
from PIL import Image
from PIL import ImageSequence
from tqdm import tqdm

# input_dirs = [
#     "workdir/navsim/iclight_vidtome_opt/navsim_16_00634_01421_lmr_0.6_gmr_0.5_alpha_t_0.0_opt",
#     "workdir/navsim/iclight_vidtome_opt/navsim_17_01352_0190_lmr_0.6_gmr_0.5_alpha_t_0.0_opt",
#     "workdir/navsim/iclight_vidtome_opt/navsim_26_03873_04225_lmr_0.6_gmr_0.5_alpha_t_0.0_opt",
#     "workdir/navsim/iclight_vidtome_opt/navsim_38_00305_00597_lmr_0.6_gmr_0.5_alpha_t_0.0_opt",
#     "workdir/navsim/iclight_vidtome_opt/navsim_50_01085_01463_lmr_0.6_gmr_0.5_alpha_t_0.0_opt"
# ]

input_dirs = "workdir/droid/iclight_vidtome_slicedit_opt"
input_dirs = [os.path.join(input_dirs, d) for d in os.listdir(input_dirs) if os.path.isdir(os.path.join(input_dirs, d))]

out_dir = "workdir/demos"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

downsample_factor = 3

for input_dir in tqdm(input_dirs, desc="Downsampling GIFs"):
    
    file_name = input_dir.split("/")[-1]

    if os.path.exists(f'{input_dir}/output.gif'):
        im = Image.open(f'{input_dir}/output.gif')
        im_gt = Image.open(f'{input_dir}/output_gt.gif')
        fps = im.info['duration'] / 1000
        resize_frames= [frame.resize((frame.width // downsample_factor, frame.height // downsample_factor)) for frame in ImageSequence.Iterator(im)]
        resize_frames_gt= [frame.resize((frame.width // downsample_factor, frame.height // downsample_factor)) for frame in ImageSequence.Iterator(im_gt)]

    elif os.path.exists(f'{input_dir}/output.mp4'):
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

        while True:
            ret, frame = cap_gt.read()
            if not ret:
                break
            im_gt.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap_gt.release()

        resize_frames = [frame.resize((frame.width // downsample_factor, frame.height // downsample_factor)) for frame in im]
        resize_frames_gt = [frame.resize((frame.width // downsample_factor, frame.height // downsample_factor)) for frame in im_gt]
    
    file_name = file_name.replace(":", "_")
    resize_frames[0].save(f"{out_dir}/{file_name}_ds{downsample_factor}.gif", save_all=True, append_images=resize_frames[1:], duration=1000/fps, loop=0)
    resize_frames_gt[0].save(f"{out_dir}/{file_name}_gt_ds{downsample_factor}.gif", save_all=True, append_images=resize_frames_gt[1:], duration=1000/fps, loop=0)

print("Done")