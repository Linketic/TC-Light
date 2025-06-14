import os
import sys
import json
import numpy as np
import torch
import torchvision.transforms as T

from scipy import misc
from tqdm import tqdm
from enum import Enum
from PIL import Image
from evaluation import eval_utils as eu

from .video_dataparser import VideoDataParser
from .sceneflow_dataparsers import read
from utils.general_utils import voxelization, process_frames
from utils.flow_utils import get_soft_mask_bwds, get_flowid


def readCamIntrinsic(file):
    with open(file, 'r') as file:
        lines = file.readlines()
    
    resolution = list(map(int, lines[1].strip().split()))
    fx, fy = map(float, lines[3].strip().split())
    cx, cy = map(float, lines[5].strip().split())
    
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ])
    
    return K

def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]
    ])

def parse_visim_file(file_path):
    extrinsics_dict = {}

    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        # skip comment lines and empty lines
        if line.startswith("#") or line.strip() == "":
            continue
        
        # parse each line
        data = line.strip().split(",")
        timestamp = int(data[0])
        position = np.array([float(data[1]), float(data[2]), float(data[3])])
        quaternion = np.array([float(data[4]), float(data[5]), float(data[6]), float(data[7])])
        
        rotation_matrix = quaternion_to_rotation_matrix(quaternion)
        
        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = rotation_matrix
        extrinsic_matrix[:3, 3] = position
        #extrinsic_matrix = np.linalg.inv(extrinsic_matrix)
        
        extrinsics_dict[timestamp] = extrinsic_matrix

    return extrinsics_dict 

def normalize(v):
    return v / np.linalg.norm(v)


class InteriorNetDataParser(VideoDataParser):

    def __init__(self, 
                 data_config,
                 device="cuda",
                 dtype=torch.float32):

        self.data_dir = "data/interiornet/HD1/3FO4K35GPEA7" if not hasattr(data_config, "data_dir") else data_config.data_dir
        self.traj = 7 if not hasattr(data_config, "traj") else data_config.traj
        self.flow_model = "memflow" if not hasattr(data_config, "flow_model") else data_config.flow_model
        self.voxel_size = None if not hasattr(data_config, "voxel_size") else data_config.voxel_size
        self.apply_mask = True if not hasattr(data_config, "apply_mask") else data_config.apply_mask
        self.contract = False if not hasattr(data_config, "contract") else data_config.contract
        self.use_raft = False if not hasattr(data_config, "use_raft") else data_config.use_raft
        self.fps = 30 if not hasattr(data_config, "fps") else data_config.fps
        self.alpha = 0.1 if not hasattr(data_config, "alpha") else data_config.alpha
        self.beta = 1e2 if not hasattr(data_config, "beta") else data_config.beta
        self.h, self.w = data_config.height, data_config.width
        self.device = device
        self.dtype = dtype
        self.unq_inv = None
        self.new_coors = None

        self.rgb_path = os.path.join(self.data_dir, f"original_{self.traj}_{self.traj}", "cam0", "data")
        self.depth_path = os.path.join(self.data_dir, f"original_{self.traj}_{self.traj}", "depth0", "data")
        self.mask_path = os.path.join(self.data_dir, f"original_{self.traj}_{self.traj}", "label0", "data")

        self.intrinsics = readCamIntrinsic(os.path.join(self.data_dir, f"velocity_angular_{self.traj}_{self.traj}", "cam0.info"))
        self.extrinsics_dict = parse_visim_file(os.path.join(self.data_dir, f"velocity_angular_{self.traj}_{self.traj}", "cam0_gt.visim"))
        self.timestamps = list(self.extrinsics_dict.keys())
        self.n_frames = len(self.timestamps)
    
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

            p_cam_homo[:, :, 1:3] *= -1

            # Transform to world coordinates
            p_world = torch.matmul(p_cam_homo, c2ws.transpose(-2, -1))[:, :, :3]  # Shape: (N, H*W, 3)

            # Reshape rgb to (N, H*W, 3)
            rgb_world = rgbs.permute(0, 2, 3, 1).reshape(N, -1, 3)  # Shape: (N, H*W, 3)
        
        return p_world, rgb_world
    
    @torch.no_grad()
    def load_video(self, frame_ids=None, rgb_threshold=0.01):
        rgbs = []
        frame_ids = frame_ids if frame_ids is not None else list(range(len(self.cam_info)))
        for i in tqdm(range(len(self.timestamps)), desc="Loading Data"):
            if i in frame_ids:
                try:
                    rgb = read(os.path.join(self.rgb_path, f"{self.timestamps[i]:019d}.png"))
                except:
                    print(f"Idx {self.timestamps[i]} is unavailable, ignored.")
                    continue

                rgbs.append(torch.tensor(rgb, dtype=self.dtype).permute(2, 0, 1))
        
        self.n_frames = len(rgbs)
        rgbs = torch.stack(rgbs, dim=0) / 255.0
        rgbs = rgbs.to(self.device)
        rgbs = process_frames(rgbs, self.h, self.w)  # Shape: (N, 3, h, w)

        return rgbs
    
    @torch.no_grad()
    def load_data(self, frame_ids=None, rgb_threshold=0.01):
        rgbs, depths, masks, c2ws = [], [], [], []
        frame_ids = frame_ids if frame_ids is not None else list(range(len(self.cam_info)))
        for i in tqdm(range(len(self.timestamps)), desc="Loading Data"):
            if i in frame_ids:
                try:
                    rgb = read(os.path.join(self.rgb_path, f"{self.timestamps[i]:019d}.png"))
                    mask = read(os.path.join(self.mask_path, f"{self.timestamps[i]:019d}_instance.png"))
                    depth = read(os.path.join(self.depth_path, f"{self.timestamps[i]:019d}.png"))
                except:
                    print(f"Idx {self.timestamps[i]} is unavailable, ignored.")
                    continue
                vs = np.array(
                    [(v - self.intrinsics[0, 2]) / self.intrinsics[0, 0] for v in range(0, depth.shape[1])])
                us = np.array(
                    [(u - self.intrinsics[1, 2]) / self.intrinsics[1, 1] for u in range(0, depth.shape[0])])
                depth = np.sqrt(np.square(depth / 1000.0) /
                    (1 + np.square(vs[np.newaxis, :]) + np.square(us[:, np.newaxis])))

                c2w = self.extrinsics_dict[self.timestamps[i]]

                rgbs.append(torch.tensor(rgb, dtype=self.dtype).permute(2, 0, 1))
                depths.append(torch.tensor(depth[None], dtype=self.dtype))
                masks.append(torch.tensor(mask[None], dtype=self.dtype))
                c2ws.append(torch.tensor(c2w, dtype=self.dtype))
        
        self.n_frames = len(rgbs)
        rgbs = torch.stack(rgbs, dim=0) / 255.0
        depths = torch.stack(depths, dim=0)
        masks = torch.stack(masks, dim=0)
        c2ws = torch.stack(c2ws, dim=0)
        N, _, H, W = rgbs.shape

        p_world, rgb_world = self.rgbd2pcd(rgbs, depths, self.intrinsics, c2ws)  # Shape: (N, H*W, 3), (N, H*W, 3)
        # from utils.general_utils import save_ply  # save to check correctness
        # save_ply(p_world.reshape(-1, 3)[::100].cpu().numpy(), rgb_world.reshape(-1, 3)[::100].cpu().numpy())

        del rgbs, depths  # Free up memory

        rgb_world = process_frames(rgb_world.reshape(N, H, W, 3).permute(0, 3, 1, 2), self.h, self.w)  # Shape: (N, 3, h, w)
        flows, past_flows, mask_bwds, _, _, _ = self.load_flow(frame_ids=frame_ids, future_flow=True, past_flow=True, gts=rgb_world)
        flow_ids = get_flowid(rgb_world, flows, mask_bwds, rgb_threshold=rgb_threshold).view(-1, 1)
        torch.cuda.empty_cache()  # Clear GPU memory
        
        rgb_world = rgb_world.permute(0, 2, 3, 1).reshape(-1, 3).to(self.device) # Shape: (N*h*w, 3)
        p_world = process_frames(p_world.reshape(N, H, W, 3).permute(0, 3, 1, 2), self.h, self.w)  # Shape: (N, 3, h, w)
        p_world = p_world.permute(0, 2, 3, 1).reshape(-1, 3).to(self.device)  # Shape: (N*h*w, 3)

        if self.apply_mask:
            masks = process_frames(masks.reshape(N, H, W, 3).permute(0, 3, 1, 2), self.h, self.w)[:, 0:1]  # Shape: (N, 1, h, w)
            masks = masks.permute(0, 2, 3, 1).reshape(-1).to(self.device)
        else:
            masks = None
        
        self.unq_inv = voxelization(flow_ids, rgb_world, p_world,
                                    self.voxel_size, instance_ids=masks,
                                    contract=self.contract)
        torch.cuda.empty_cache()  # Clear GPU memory

        return rgb_world, p_world, c2ws, flows, past_flows, mask_bwds
    
    

