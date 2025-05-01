import torch
import clip

import sys
import yaml

import shutil
import os
import glob
import argparse
import numpy as np

from omegaconf import OmegaConf, DictConfig
from collections import defaultdict
from transformers import AutoProcessor, AutoModel

from skimage.metrics import structural_similarity

import eval_utils as eu

def yaml_load(path):
    with open(path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='/data1/yang_liu/python_workspace/IC-Light/workdir/realistic')
    parser.add_argument('--eval_cost', action='store_true')
    args = parser.parse_args()

    st = 50
    eval_cost = args.eval_cost
    output_dir = args.output_dir

    config_dict = yaml_load(os.path.join(output_dir, 'config.yaml'))

    try:
        config = OmegaConf.load(os.path.join(output_dir, 'config.yaml'))
        if config.data.scene_type.lower() == "sceneflow":
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from utils.dataparsers import SceneFlowDataParser
            data_parser = SceneFlowDataParser(config.data, config.device)
            data_parser.load_video_flow(eu.get_frame_ids(config.generation.frame_range, config.generation.frame_ids), past_flow=True);
            flow_fwd_list = data_parser.flows
            flow_bwd_list = data_parser.past_flows
            print(f"Loaded optical flow from {config.data.scene_type}")
        else:
            raise NotImplementedError(f"Scene type {config.data.scene_type} is not supported.")
    except:
        flow_fwd_list=None
        flow_bwd_list=None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    pick_model = AutoModel.from_pretrained("pickapic-anonymous/PickScore_v1").to(device)
    pick_processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    raft_model = eu.prepare_raft_model(device)

    rearrange = lambda x: (np.array(x)/255).reshape(-1,1)
    l2_norm = lambda x,y: np.linalg.norm(rearrange(x)-rearrange(y))/rearrange(x).shape[0]
    l1_norm = lambda x,y: np.linalg.norm(rearrange(x)-rearrange(y), ord=1)/rearrange(x).shape[0]
    
    main_dict = {
        'iclight_vid': {},
    }
    
    scores_main = defaultdict(float)
    result_file_path = f'{output_dir}/result.txt'
    with open(result_file_path, 'w') as f:

        video_name = config_dict['input_path'].split('/')[-2]
            
        for key, prompt in config_dict['generation']['prompt'].items():

            for k in main_dict.keys():

                main_dict[k][video_name] = {}
                scores = scores_main.copy()
                if os.path.exists(f'{output_dir}/output_opt.mp4'):
                    video_path = f'{output_dir}/output_opt.mp4'
                else:
                    video_path = f'{output_dir}/output.mp4'
                source_video_path = f'{output_dir}/output_gt.mp4'

                if os.path.exists(video_path):
                    pil_list = eu.video_to_pil_list(video_path)
                    source_pil_list = eu.video_to_pil_list(source_video_path)

                    # check if the source image share the same shape with video, if not, reshape the source
                    if pil_list[0].size != source_pil_list[0].size:
                        source_pil_list = [im.resize(pil_list[0].size) for im in source_pil_list]
                    
                    scores['clip-frame'] = eu.clip_frame(pil_list, preprocess, device, model)
                    try:
                        scores['clip-text'] = eu.clip_text(pil_list, prompt, preprocess, device, model)
                    except:
                        scores['clip-text'] = 0
                        print(f'Error in clip-text for {video_name} - {prompt}')
                    
                    scores['pick-score'] = eu.pick_score_func(pil_list, prompt, pick_model, pick_processor, device)
                    
                    # scores['lpips-frame'] = eu.FrameLPIPS(pil_list, source_pil_list, device)

                    scores['warp-error-ssim'] = eu.SaveWarpingImage(pil_list, source_pil_list, raft_model, device, 
                                                                    structural_similarity, flow_fwd_list, flow_bwd_list)
                    # print(f'{video_name} - {prompt} - {k} - ', end='\n')

                    if eval_cost:
                        config_path = f'{output_dir}/{video_name}/{prompt}/config.yaml'
                        # z to move the cost to the end of the dictionary
                        
                        if 'sec_per_frame' in config_dict.keys():
                            scores['z_fps'] = 1 / config_dict['sec_per_frame']
                        else:
                            scores['z_fps'] = config_dict['frame_per_sec']
                        scores['z_max_memory_allocated(M)'] = config_dict['max_memory_allocated']
                        scores['z_resolution'] = np.sqrt(pil_list[0].size[0]*pil_list[0].size[1])
                        scores['z_total_frames'] = int(config_dict['total_number_of_frames'])
                        scores['z_total_time(s)'] = config_dict['total_time']

                main_dict[k][video_name][prompt] = scores.copy()
        
        print(f'{video_name} - {prompt} - ', end='\n')
        f.write(f'{video_name} - {prompt}\n')
        for k in main_dict.keys():
            print(f'\t{k}: ', end='')
            for s in sorted(main_dict[k][video_name][prompt].keys()):
                if 'warp-error-l1' in s:
                    print(f'{(main_dict[k][video_name][prompt][s]*100000):.2f}', end=', ')
                    f.write(f'{s}: {(main_dict[k][video_name][prompt][s]*100000):.2f}\n')
                elif 'warp-error-l2' in s or 'warp-error-ssim' in s:
                    print(f'{(main_dict[k][video_name][prompt][s]*100):.2f}', end=', ')
                    f.write(f'{s}: {(main_dict[k][video_name][prompt][s]*100):.2f}\n')
                else:
                    print(f'{main_dict[k][video_name][prompt][s]:.4f}', end=', ')
                    f.write(f'{s}: {main_dict[k][video_name][prompt][s]:.4f}\n')
            print()
        print()
