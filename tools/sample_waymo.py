import os
import json
import argparse
import random
import numpy as np
from tqdm import tqdm

# rainy: 492, night:214, 642, dawn:200, cloudy: 253, 184, sunny: 648, 287
target_idx = [492, 214, 642, 200, 253, 184, 648, 287]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", help="input folder containing images", default="data/waymo_org/kitti_format")
    parser.add_argument("-o", "--output_folder", help="output folder", default=None)
    args = parser.parse_args()

    if args.output_folder is None:
        args.output_folder = args.input_folder.replace("/waymo_", "/waymo_sampled_")
    
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        if not os.path.exists(os.path.join(args.output_folder, "training/image_0")):
            os.makedirs(os.path.join(args.output_folder, "training/image_0"))

    info = np.load(os.path.join(args.input_folder, "waymo_infos_train.pkl"), allow_pickle=True)

    info_seq = {}
    # info[i]['image']['image_idx'] is of format {a}{bbb}{ccc}, a distinguishes train/val/test, 
    # bbb is the sequence number, ccc is the frame number
    for i in range(len(info)):
        seq_id = f"{info[i]['image']['image_idx']:07d}"[1:4]
        if seq_id not in info_seq:
            info_seq[seq_id] = []
        info_seq[seq_id].append(info[i])
    print(f"{len(info_seq)} sequences contained in waymo_infos_train.pkl")

    info_seq_selected = {}
    for idx in target_idx:
        seq_id = f"{idx:03d}"
        if seq_id not in info_seq:
            print(f"Sequence {seq_id} not found in waymo_infos_train.pkl")
            continue
        
        # divide different sequences into different folders
        if not os.path.exists(os.path.join(args.output_folder, "training/image_0", seq_id)):
            os.makedirs(os.path.join(args.output_folder, "training/image_0", seq_id))
        
        for i in tqdm(range(len(info_seq[seq_id])), desc=f"Copying images for sequence {seq_id}"):
            img_path = os.path.join(args.input_folder, info_seq[seq_id][i]['image']['image_path'])
            out_data_path = os.path.join(seq_id, info_seq[seq_id][i]['image']['image_path'].split("/")[-1])
            os.system(f"cp {img_path} {os.path.join(args.output_folder, 'training/image_0', out_data_path)}")
        
        info_seq_selected[seq_id] = info_seq[seq_id]
    
    # save selected sequences to a new pkl file
    with open(os.path.join(args.output_folder, "waymo_infos_train.pkl"), 'wb') as f:
        np.save(f, info_seq_selected)
    
    print(f"Sampled data saved to {args.output_folder}")