import os
import json
import argparse
import random
import numpy as np
from tqdm import tqdm

random.seed(123)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", help="input folder containing images", default="data/navsim_org")
    parser.add_argument("-o", "--output_folder", help="output folder", default=None)
    parser.add_argument("-ns", "--num_scenes", help="number of scenes to be randomly selected", default=5)
    parser.add_argument("-nt", "--num_threshold", help="scenes with images more than this threshold will be considered", default=150)
    args = parser.parse_args()

    if args.output_folder is None:
        args.output_folder = args.input_folder + "_sampled"
    
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    
    input_img_dir = os.path.join(args.input_folder, "sensor_blobs", "trainval")
    input_log_dir = os.path.join(args.input_folder, "navsim_logs", "trainval")
    available_scenes = []
    for file in os.listdir(input_img_dir):
        if len(os.listdir(os.path.join(input_img_dir, file, "CAM_F0"))) >= args.num_threshold:
            available_scenes.append(file)
    available_scenes = sorted(available_scenes)

    # randomly select scenes
    selected_scenes = random.sample(available_scenes, args.num_scenes)
    print(f"Selected scenes: {selected_scenes}")

    for scene in selected_scenes:
        output_img_dir = os.path.join(args.output_folder, "sensor_blobs", "trainval")
        if not os.path.exists(os.path.join(output_img_dir, scene)):
            os.makedirs(os.path.join(output_img_dir, scene))
            os.makedirs(os.path.join(output_img_dir, scene, "CAM_B0"))
            os.makedirs(os.path.join(output_img_dir, scene, "CAM_F0"))
            os.makedirs(os.path.join(output_img_dir, scene, "CAM_L0"))
            os.makedirs(os.path.join(output_img_dir, scene, "CAM_L1"))
            os.makedirs(os.path.join(output_img_dir, scene, "CAM_L2"))
            os.makedirs(os.path.join(output_img_dir, scene, "CAM_R0"))
            os.makedirs(os.path.join(output_img_dir, scene, "CAM_R1"))
            os.makedirs(os.path.join(output_img_dir, scene, "CAM_R2"))
        
        output_log_dir = os.path.join(args.output_folder, "navsim_logs", "trainval")
        if not os.path.exists(os.path.join(output_log_dir)):
            os.makedirs(os.path.join(output_log_dir))
        
        info = np.load(os.path.join(input_log_dir, f"{scene}.pkl"), allow_pickle=True)
        info_available = [info[i] for i in range(len(info)) if os.path.exists(os.path.join(input_img_dir, info[i]['cams']["CAM_F0"]['data_path']))]
        with open(os.path.join(output_log_dir, f"{scene}.pkl"), 'wb') as f:
            np.save(f, info_available)

        # copy images
        for i in tqdm(range(len(info_available)), desc=f"Copying images for scene {scene}"):
            for cam_type in info_available[i]['cams'].keys():
                img_path = os.path.join(input_img_dir, info_available[i]['cams'][cam_type]['data_path'])
                out_data_path = info_available[i]['cams'][cam_type]['data_path'].replace(info_available[i]['cams'][cam_type]['data_path'].split("/")[-1].split('.')[0], f"{i:04d}")
                os.system(f"cp {img_path} {os.path.join(output_img_dir, out_data_path)}")
    
    print(f"Sampled data saved to {args.output_folder}")