import os
import torch
import torch_scatter
import argparse
import numpy as np

import evaluation.eval_utils as eu
from omegaconf import OmegaConf
from tqdm import tqdm

@torch.no_grad()
def get_recon_metrics(unq_inv, data_path, device, dtype):
        pil_list = eu.video_to_pil_list(data_path)
        edited_images = torch.concat([eu.load_image(pil, device, dtype) for pil in pil_list], dim=0) / 255.0
        N, _, H, W = edited_images.shape
        pil_tensor = edited_images.permute(0, 2, 3, 1).reshape(N*H*W, -1)
        pil_tensor = torch_scatter.scatter(pil_tensor, unq_inv, dim=0, reduce='mean')

        images = pil_tensor[unq_inv].reshape(N, H*W, -1) # N x HW x 3
        images = torch.clamp(images, 0, 1).reshape(N, H, W, 3).permute(0, 3, 1, 2)  # N x 3 x H x W

        ssims = []
        psnrs = []
        lpipss = []

        for i in tqdm(range(len(images))):
            psnr = eu.psnr(images[i], edited_images[i]).mean()
            ssim = eu.ssim(images[i], edited_images[i])
            lpips = eu.lpips_func(images[i], edited_images[i])

            # if psnr is inf, skip it
            if torch.isinf(psnr):
                continue

            psnrs.append(psnr)
            ssims.append(ssim)
            lpipss.append(lpips)
        
        return {
            'SSIM': torch.tensor(ssims).mean(),
            'PSNR': torch.tensor(psnrs).mean(),
            'LPIPS': torch.tensor(lpipss).mean()
        }

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--eval_output', action='store_true')
    args = parser.parse_args()

    config = OmegaConf.load(os.path.join(args.output_dir, 'config.yaml'))
    data_config = config.data
    if data_config.scene_type.lower() == "sceneflow":
        from utils.dataparsers import SceneFlowDataParser
        data_parser = SceneFlowDataParser(data_config, config.device)
    elif data_config.scene_type.lower() == "carla":
        from utils.dataparsers import CarlaDataParser
        data_parser = CarlaDataParser(data_config, config.device)
    elif data_config.scene_type.lower() == "robotrix":
        from utils.dataparsers import RobotrixDataParser
        data_parser = RobotrixDataParser(data_config, config.device)
    elif data_config.scene_type.lower() == "interiornet":
        from utils.dataparsers import InteriorNetDataParser
        data_parser = InteriorNetDataParser(data_config, config.device)
    elif data_config.scene_type.lower() == "video":
        from utils.dataparsers import VideoDataParser
        data_parser = VideoDataParser(data_config, config.device)
    else:
        raise NotImplementedError(f"Scene type {data_config.scene_type} is not supported.")
    
    frame_ids = eu.get_frame_ids(config.generation.frame_range, config.generation.frame_ids)
    frames, _, _, flows, past_flows, mask_bwds = data_parser.load_video(frame_ids=frame_ids)
    
    result_file_path = f'{args.output_dir}/result_recon.txt'
    with open(result_file_path, 'w') as f:
        f.write(f'#Pixels: {frames.shape[0] * frames.shape[2] * frames.shape[3]}\n')
        f.write(f'#UVT: {torch.unique(data_parser.unq_inv).shape[0]}\n')

        output = get_recon_metrics(data_parser.unq_inv, os.path.join(args.output_dir, "output_gt.mp4"), flows.device, flows.dtype)
        f.write(f'SSIM(org): {output["SSIM"]:.4f}\n')
        f.write(f'PSNR(org): {output["PSNR"]:.4f}\n')
        f.write(f'LPIPS(org): {output["LPIPS"]:.4f}\n')

        if args.eval_output:
            output = get_recon_metrics(data_parser.unq_inv, os.path.join(args.output_dir, "output.mp4"), flows.device, flows.dtype)
            f.write(f'SSIM(edited): {output["SSIM"]:.4f}\n')
            f.write(f'PSNR(edited): {output["PSNR"]:.4f}\n')
            f.write(f'LPIPS(edited): {output["LPIPS"]:.4f}\n')
    
    print(f"Reconstruction metrics saved to {result_file_path}")