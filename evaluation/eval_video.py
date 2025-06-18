import os
import sys
import yaml
import torch
import argparse
import numpy as np

import clip
from collections import defaultdict
from datetime import datetime
from omegaconf import OmegaConf
from transformers import AutoProcessor, AutoModel
from skimage.metrics import structural_similarity

import eval_utils as eu

def evaluate_video(video_name, prompt, config, device, output_dir, eval_cost,
                   model, preprocess, pick_model, pick_processor,
                   raft_model, flow_fwd_list, flow_bwd_list, main_dict):

    for k in main_dict.keys():
        main_dict[k][video_name] = {}
        scores = defaultdict(float)

        video_path = f'{output_dir}/output_opt.mp4' if os.path.exists(f'{output_dir}/output_opt.mp4') else f'{output_dir}/output.mp4'
        source_video_path = f'{output_dir}/output_gt.mp4'

        if not os.path.exists(video_path):
            continue

        pil_list = eu.video_to_pil_list(video_path)
        source_pil_list = eu.video_to_pil_list(source_video_path)

        # Resize source frames if necessary
        if pil_list[0].size != source_pil_list[0].size:
            source_pil_list = [im.resize(pil_list[0].size) for im in source_pil_list]

        # Run metrics
        scores['clip-frame'] = eu.clip_frame(pil_list, preprocess, device, model)

        try:
            scores['clip-text'] = eu.clip_text(pil_list, prompt, preprocess, device, model)
        except Exception:
            print(f"[WARN] Prompt too long: '{prompt}', splitting.")
            scores_list = [
                eu.clip_text(pil_list, p, preprocess, device, model)
                for p in prompt.split('.') if p.strip()
            ]
            scores['clip-text'] = np.mean(scores_list)

        scores['pick-score'] = eu.pick_score_func(pil_list, prompt, pick_model, pick_processor, device)

        scores['warp-error-ssim'] = eu.SaveWarpingImage(
            pil_list, source_pil_list, raft_model, device,
            structural_similarity, flow_fwd_list, flow_bwd_list
        )

        if eval_cost:
            scores['z_fps'] = 1 / config.sec_per_frame
            scores['z_max_memory_allocated(M)'] = config.max_memory_allocated
            scores['z_resolution'] = np.sqrt(pil_list[0].size[0] * pil_list[0].size[1])
            scores['z_total_frames'] = config.total_number_of_frames
            scores['z_total_time(s)'] = config.total_time

        main_dict[k][video_name][prompt] = scores.copy()

def print_and_save_results(video_name, prompt, main_dict, output_path):
    result_file_path = os.path.join(output_path, 'result.txt')
    with open(result_file_path, 'w') as f:
        print(f"{video_name} - {prompt}")
        f.write(f"{video_name} - {prompt}\n")

        for k in main_dict:
            print(f"\t{k}: ", end='')
            for metric, score in sorted(main_dict[k][video_name][prompt].items()):
                if 'warp-error-l1' in metric:
                    value = score * 1e5
                    print(f"{value:.2f}", end=', ')
                    f.write(f"{metric}: {value:.2f}\n")
                elif 'warp-error-l2' in metric or 'warp-error-ssim' in metric:
                    value = score * 100
                    print(f"{value:.2f}", end=', ')
                    f.write(f"{metric}: {value:.2f}\n")
                else:
                    print(f"{score:.4f}", end=', ')
                    f.write(f"{metric}: {score:.4f}\n")
            print()
        print()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='workdir')
    parser.add_argument('--eval_cost', action='store_true')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = args.output_dir
    eval_cost = args.eval_cost

    config = OmegaConf.load(os.path.join(output_dir, 'config.yaml'))

    try:
        if config.data.scene_type.lower() == "sceneflow":
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from utils.dataparsers import SceneFlowDataParser
            data_parser = SceneFlowDataParser(config.data, config.device)
            data_parser.load_flow(eu.get_frame_ids(config.generation.frame_range, config.generation.frame_ids), past_flow=True);
            flow_fwd_list = data_parser.flows
            flow_bwd_list = data_parser.past_flows
            print(f"Loaded optical flow from {config.data.scene_type}")
        else:
            raise NotImplementedError(f"Scene type {config.data.scene_type} is not supported.")
    except:
        flow_fwd_list=None
        flow_bwd_list=None

    # Load models
    model, preprocess = clip.load("ViT-B/32", device=device)
    pick_model = AutoModel.from_pretrained("pickapic-anonymous/PickScore_v1").to(device)
    pick_processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    raft_model = eu.prepare_raft_model(device)

    main_dict = {'tclight': {}}
    video_name = config.input_path.split('/')[-2] if hasattr(config, "input_path") else 'unknown_video'

    for key, prompt in config.generation.prompt.items():
        evaluate_video(
            video_name, prompt, config, device, output_dir, eval_cost,
            model, preprocess, pick_model, pick_processor,
            raft_model, flow_fwd_list, flow_bwd_list, main_dict
        )
        print_and_save_results(video_name, prompt, main_dict, output_dir)
