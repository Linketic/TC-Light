import os
import sys
import json
import numpy as np
import torch
import cv2
import torchvision.transforms as T

from scipy import misc
from tqdm import tqdm
from enum import Enum
from PIL import Image
from evaluation import eval_utils as eu

from .video_dataparser import VideoDataParser
from utils.general_utils import voxelization, process_frames
from utils.flow_utils import get_mask_bwds, get_flowid

class CarlaDataParser(VideoDataParser):

    def __init__(self, 
                 data_config,
                 device="cuda",
                 dtype=torch.float32):

        self.data_dir = "data/carla/data_collect_town01_results" if not hasattr(data_config, "data_dir") else data_config.data_dir
        self.scene_name = "routes_town01_02_06_20_36_50" if not hasattr(data_config, "scene_name") else data_config.scene_name
        self.flow_model = "memflow" if not hasattr(data_config, "flow_model") else data_config.memflow
        self.fov = 90 if not hasattr(data_config, "fov") else data_config.fov  # in degrees
        self.x_shift = 1.5 if not hasattr(data_config, "x_shift") else data_config.x_shift
        self.y_shift = 0.0 if not hasattr(data_config, "y_shift") else data_config.y_shift
        self.z_shift = 2.5 if not hasattr(data_config, "z_shift") else data_config.z_shift
        self.voxel_size = None if not hasattr(data_config, "voxel_size") else data_config.voxel_size
        self.contract = False if not hasattr(data_config, "contract") else data_config.contract
        self.fps = 30 if not hasattr(data_config, "fps") else data_config.fps
        self.alpha = 0.1 if not hasattr(data_config, "alpha") else data_config.alpha
        self.h, self.w = data_config.height, data_config.width
        self.device = device
        self.dtype = dtype
        self.unq_inv = None
        self.new_coors = None

        self.rgb_path = os.path.join(self.data_dir, self.scene_name, "rgb_front")
        self.depth_path = os.path.join(self.data_dir, self.scene_name, "depth_front")
        self.mask_path = os.path.join(self.data_dir, self.scene_name, "sem_seg_front")
        self.extrinsic_path = os.path.join(self.data_dir, self.scene_name, "ego_trans_matrix")
        
        self.n_frames = len(os.listdir(self.extrinsic_path))
    
    def rgbd2pcd(self, rgbs, depths, intrinsics, c2ws):
        # Assuming rgbs is of shape (N, 3, H, W), depths is of shape (N, 1, H, W), and c2ws is of shape (N, 4, 4)
        N, _, H, W = rgbs.shape
        if len(intrinsics.shape) == 2:
            intrinsics = intrinsics[None]
        intrinsics = torch.tensor(intrinsics, dtype=torch.float32, device=rgbs.device)

        with torch.no_grad():
            # Create meshgrid for x and y coordinates
            pos_x, pos_y = torch.meshgrid(torch.arange(W, device=rgbs.device), torch.arange(H, device=rgbs.device), indexing='xy')
            pos_x = pos_x.unsqueeze(0).expand(N, -1, -1)  # Shape: (N, H, W)
            pos_y = pos_y.unsqueeze(0).expand(N, -1, -1)  # Shape: (N, H, W)

            # Stack x and y coordinates and reshape to (N, H*W, 2)
            p_img = torch.stack([pos_x, pos_y], dim=-1).reshape(N, -1, 2)  # Shape: (N, H*W, 2)

            # Compute x_cam and y_cam
            x_cam = (p_img[:, :, 0] - intrinsics[:, 0, 2].unsqueeze(1)) * depths.reshape(N, -1) / intrinsics[:, 0, 0].unsqueeze(1)
            y_cam = (p_img[:, :, 1] - intrinsics[:, 1, 2].unsqueeze(1)) * depths.reshape(N, -1) / intrinsics[:, 1, 1].unsqueeze(1)

            # Stack x_cam, y_cam, depth, and ones to form homogeneous coordinates
            p_cam_homo = torch.stack([x_cam, y_cam, depths.reshape(N, -1), torch.ones_like(x_cam, device=rgbs.device)], dim=-1)  # Shape: (N, H*W, 4)
            p_cam_homo = p_cam_homo[:, :, [2, 0, 1, 3]]
            p_cam_homo[:, 1:3] *= -1

            # Transform to world coordinates
            p_world = torch.matmul(p_cam_homo, c2ws.transpose(-2, -1))[:, :, :3]  # Shape: (N, H*W, 3)

            # Reshape rgb to (N, H*W, 3)
            rgb_world = rgbs.permute(0, 2, 3, 1).reshape(N, -1, 3)  # Shape: (N, H*W, 3)
        
        return p_world, rgb_world
    
    @torch.no_grad()
    def load_video(self, frame_ids=None, rgb_threshold=0.01):
        rgbs, depths, masks, c2ws = [], [], [], []
        frame_ids = frame_ids if frame_ids is not None else list(range(self.n_frames))
        for i in tqdm(range(self.n_frames), desc="Loading Data"):
            if i in frame_ids:
                rgb = cv2.imread(os.path.join(self.rgb_path, f"{i:04d}.png"))
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                mask = cv2.imread(os.path.join(self.mask_path, f"{i:04d}.png"))
                depth = cv2.imread(os.path.join(self.depth_path, f"{i:04d}.png"))
                depth = (depth[:, :, 2] + depth[:, :, 1] * 256.0 + depth[:, :, 0] * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1)
                depth = depth * 1000.0  # in meters

                with open(os.path.join(self.extrinsic_path, f"{i:04d}.json")) as f:
                    c2w = np.array(json.load(f))
                    c2w[0, 3] += self.x_shift
                    c2w[1, 3] += self.y_shift
                    c2w[2, 3] += self.z_shift

                rgbs.append(torch.tensor(rgb, dtype=self.dtype, device=self.device).permute(2, 0, 1))
                depths.append(torch.tensor(depth[None], dtype=self.dtype, device=self.device))
                masks.append(torch.tensor(mask, dtype=self.dtype, device=self.device).permute(2, 0, 1))
                c2ws.append(torch.tensor(c2w, dtype=self.dtype, device=self.device))
        
        self.n_frames = len(rgbs)
        rgbs = torch.stack(rgbs, dim=0) / 255.0
        depths = torch.stack(depths, dim=0)
        c2ws = torch.stack(c2ws, dim=0)
        N, _, H, W = rgbs.shape

        f = W / (2 * np.tan(np.deg2rad(self.fov/2)))
        intrinsics = np.array([[f, 0, W/2], [0, f, H/2], [0, 0, 1]])

        p_world, rgb_world = self.rgbd2pcd(rgbs, depths, intrinsics, c2ws)  # Shape: (N, H*W, 3), (N, H*W, 3)
        p_world = process_frames(p_world.reshape(N, H, W, 3).permute(0, 3, 1, 2), self.h, self.w)  # Shape: (N, 3, h, w)
        rgb_world = process_frames(rgb_world.reshape(N, H, W, 3).permute(0, 3, 1, 2), self.h, self.w)  # Shape: (N, 3, h, w)
        flows, past_flows, mask_bwds = self.load_flow(frame_ids=frame_ids, future_flow=True, past_flow=True, gts=rgb_world)
        flow_ids = get_flowid(rgb_world, flows, mask_bwds, rgb_threshold=rgb_threshold)

        del rgbs, depths  # Free up memory

        # from utils.general_utils import save_ply  # save to check correctness
        # save_ply(p_world.reshape(-1, 3)[::100].cpu().numpy(), rgb_world.reshape(-1, 3)[::100].cpu().numpy())

        self.unq_inv = voxelization(flow_ids.reshape(-1), 
                                    rgb_world.permute(0, 2, 3, 1).reshape(-1, 3), 
                                    p_world.permute(0, 2, 3, 1).reshape(-1, 3),
                                    self.voxel_size, contract=self.contract)

        return rgb_world, p_world, c2ws, flows, past_flows, mask_bwds
    
    

