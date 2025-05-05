import torch
import clip

import yaml

import os
import argparse
import numpy as np

import eval_utils as eu

def yaml_load(path):
    with open(path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    st = 50                
    output_dir = args.output_dir

    config_dict = yaml_load(os.path.join(output_dir, 'config.yaml'))
    with open(os.path.join(output_dir, 'result.txt'), 'r') as f:
        lines_result = f.readlines()

    # convert lines_result[2] to dict, the string is like 'clip-text: 0.0000\n'
    clip_t_result = lines_result[2].strip().split(': ')
    clip_t = float(clip_t_result[1])
    if clip_t == 0:
        print(f'The clip-t of {output_dir} is 0, reevaluate it')
        prompt = list(config_dict['generation']['prompt'].values())[0]
        prompt_list, scores_list = prompt.split('.'), []
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        pil_list = eu.video_to_pil_list(os.path.join(output_dir, 'output.mp4'))
        for p in prompt_list:
            if p != '':
                scores_list.append(eu.clip_text(pil_list, p, preprocess, device, model))
        clip_t = np.mean(scores_list)

        # write the new clip-t to the result.txt
        lines_result[2] = f'clip-text: {clip_t:.4f}\n'
        with open(os.path.join(output_dir, 'result.txt'), 'w') as f:
            f.writelines(lines_result)
        
        print(f'Update the clip-t of {output_dir} to {clip_t:.4f}')