import os
import json
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", help="input folder containing images", default="data/robotrix/000_hamburghaus/000")
    parser.add_argument("-o", "--output_folder", help="output folder", default=None)
    parser.add_argument("-fb", "--frame_begin", help="begin frame of sampled data", default=500)
    parser.add_argument("-fe", "--frame_end", help="end frame of sampled data", default=1500)
    parser.add_argument("-fs", "--frame_stride", help="stride of sampled data", default=5)
    args = parser.parse_args()

    if args.output_folder is None:
        args.output_folder = args.input_folder + "_sampled"
    
    data_info_name = args.input_folder.split("/")[-2].split("_")[1] + "_" + args.input_folder.split("/")[-1]
    with open(os.path.join(args.input_folder, f'{data_info_name}.json')) as f:
        data_info = json.load(f)
        data_info_sampled = data_info.copy()
        data_info_sampled['frames'] = []
    
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        os.makedirs(os.path.join(args.output_folder, "rgb"))
        os.makedirs(os.path.join(args.output_folder, "depth"))
        os.makedirs(os.path.join(args.output_folder, "mask"))
        for camera in data_info['cameras']:
            os.makedirs(os.path.join(args.output_folder, "rgb", camera['name']))
            os.makedirs(os.path.join(args.output_folder, "depth", camera['name']))
            os.makedirs(os.path.join(args.output_folder, "mask", camera['name']))
    
    # copy sceneObject.json
    if not os.path.exists(f"{args.output_folder}/sceneObject.json"):
        os.system(f"cp {args.input_folder}/sceneObject.json {args.output_folder}/sceneObject.json")
    
    for i in tqdm(range(int(args.frame_begin), int(args.frame_end), int(args.frame_stride)), desc="Sampling Data"):
        data_info_sampled['frames'].append(data_info['frames'][i])
        cam_id = int(data_info['frames'][i]['id'])
        for camera in data_info['cameras']:
            if not os.path.exists(os.path.join(args.output_folder, "rgb", camera['name'], f"{cam_id+1:06d}.jpg")):
                os.system(f"cp {args.input_folder}/rgb/{camera['name']}/{cam_id+1:06d}.jpg {args.output_folder}/rgb/{camera['name']}/{cam_id+1:06d}.jpg")
            
            if not os.path.exists(os.path.join(args.output_folder, "depth", camera['name'], f"{cam_id+1:06d}.png")):
                os.system(f"cp {args.input_folder}/depth/{camera['name']}/{cam_id+1:06d}.png {args.output_folder}/depth/{camera['name']}/{cam_id+1:06d}.png")
            
            if not os.path.exists(os.path.join(args.output_folder, "mask", camera['name'], f"{cam_id+1:06d}.png")):
                os.system(f"cp {args.input_folder}/mask/{camera['name']}/{cam_id+1:06d}.png {args.output_folder}/mask/{camera['name']}/{cam_id+1:06d}.png")
        
    with open(os.path.join(args.output_folder, f'{data_info_name}.json'), 'w') as f:
        json.dump(data_info_sampled, f, indent=4)
    
    print(f"Sampled data saved to {args.output_folder}")