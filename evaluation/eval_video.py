import torch
import clip

import sys
import yaml

import shutil
import os
import glob
import argparse
import numpy as np


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
    parser.add_argument('--prepare', type=bool, default=False)
    parser.add_argument('--eval_cost', type=bool, default=False)
    args = parser.parse_args()

    st = 50
    prepare = args.prepare
    eval_cost = args.eval_cost
    output_dir = args.output_dir

    config_dict = yaml_load(os.path.join(output_dir, 'config.yaml'))

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
                video_path = f'{output_dir}/output.gif'
                source_video_path = f'{output_dir}/output_gt.gif'

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
                    if k == 'rerender':
                        # scores['warp-error-l1'] = eu.warp_video(pil_list, source_pil_list[1:-1], raft_model, device, l2_norm)
                        # scores['warp-error-l2'] = eu.warp_video(pil_list, source_pil_list[1:-1], raft_model, device, l1_norm)
                        scores['warp-error-ssim'] = eu.warp_video(pil_list, source_pil_list[1:-1], raft_model, device, structural_similarity)
                    else:
                        # scores['warp-error-l1'] = eu.warp_video(pil_list, source_pil_list, raft_model, device, l2_norm)
                        # scores['warp-error-l2'] = eu.warp_video(pil_list, source_pil_list, raft_model, device, l1_norm)
                        scores['warp-error-ssim'] = eu.SaveWarpingImage(pil_list, source_pil_list, raft_model, device, structural_similarity)
                    # print(f'{video_name} - {prompt} - {k} - ', end='\n')

                    if eval_cost:
                        config_path = f'{output_dir}/{video_name}/{prompt}/config.yaml'
                        with open(config_path, 'r') as cf:
                            config = yaml.load(cf, Loader=yaml.FullLoader)
                        # z to move the cost to the end of the dictionary
                        
                        if 'sec_per_frame' in config.keys():
                            scores['z_fps'] = 1 / config['sec_per_frame']
                        else:
                            scores['z_fps'] = config['frame_per_sec']
                        scores['z_max_memory_allocated'] = config['max_memory_allocated']
                        scores['z_resolution'] = np.sqrt(pil_list[0].size[0]*pil_list[0].size[1])
                        scores['z_total_frames'] = config['total_number_of_frames']
                        scores['z_total_time'] = config['total_time']

                main_dict[k][video_name][prompt] = scores.copy()
        
        print(f'{video_name} - {prompt} - ', end='\n')
        f.write(f'{video_name} - {prompt}\n')
        for k in main_dict.keys():
            print(f'\t{k}: ', end='')
            for s in sorted(main_dict[k][video_name][prompt].keys()):
                if 'warp-error-l1' in s:
                    print(f'{(main_dict[k][video_name][prompt][s]*100000):.2f}', end=', ')
                    f.write(f'{(main_dict[k][video_name][prompt][s]*100000):.2f}\n')
                elif 'warp-error-l2' in s or 'warp-error-ssim' in s:
                    print(f'{(main_dict[k][video_name][prompt][s]*100):.2f}', end=', ')
                    f.write(f'{(main_dict[k][video_name][prompt][s]*100):.2f}\n')
                else:
                    print(f'{main_dict[k][video_name][prompt][s]:.4f}', end=', ')
                    f.write(f'{main_dict[k][video_name][prompt][s]:.4f}, ')
            print()
        print()

        
        for k in main_dict.keys():
            samp_num = 0
            scores = scores_main.copy()
            for video_name in main_dict[k]:
                for prompt in main_dict[k][video_name]:
                    for score in main_dict[k][video_name][prompt]:
                        scores[score] += main_dict[k][video_name][prompt][score]
                    samp_num += 1
            for score in scores:
                scores[score] /= samp_num
            f.write(f'{k} - {scores}\n')
            print(k,scores)
