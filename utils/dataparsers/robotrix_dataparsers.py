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
from utils.general_utils import voxelization, process_frames
from utils.flow_utils import get_soft_mask_bwds, get_flowid

CamType = Enum("CamType", ("FirstPersonCamera", "LeftHandCamera", "RightHandCamera", "MainRoomCamera", "SecondarRoomCamera"))

def ypr2rotmat(yaw, pitch, roll):
    yaw = np.deg2rad(yaw)
    pitch = np.deg2rad(pitch)
    roll = np.deg2rad(roll)
    
    yaw_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                           [np.sin(yaw), np.cos(yaw), 0],
                           [0, 0, 1]])
    pitch_matrix = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                             [0, 1, 0],
                             [-np.sin(pitch), 0, np.cos(pitch)]])
    roll_matrix = np.array([[1, 0, 0],
                            [0, np.cos(roll), -np.sin(roll)],
                            [0, np.sin(roll), np.cos(roll)]])
    
    return yaw_matrix @ pitch_matrix @ roll_matrix

class RobotrixDataParser(VideoDataParser):

    def __init__(self, 
                 data_config,
                 device="cuda",
                 dtype=torch.float32):

        self.data_dir = "data/robotrix" if not hasattr(data_config, "data_dir") else data_config.data_dir
        self.scene_name = "000_hamburghaus" if not hasattr(data_config, "scene_name") else data_config.scene_name
        self.scene_id = 0 if not hasattr(data_config, "scene_id") else data_config.scene_id
        self.sampled = True if not hasattr(data_config, "sampled") else data_config.sampled
        self.cam_type = "FirstPersonCamera" if not hasattr(data_config, "cam_type") else data_config.cam_type
        self.flow_model = "memflow" if not hasattr(data_config, "flow_model") else data_config.flow_model
        self.voxel_size = None if not hasattr(data_config, "voxel_size") else data_config.voxel_size
        self.contract = False if not hasattr(data_config, "contract") else data_config.contract
        self.use_raft = False if not hasattr(data_config, "use_raft") else data_config.use_raft
        self.fps = 30 if not hasattr(data_config, "fps") else data_config.fps
        self.alpha = 0.1 if not hasattr(data_config, "alpha") else data_config.alpha
        self.h, self.w = data_config.height, data_config.width
        self.device = device
        self.dtype = dtype
        self.unq_inv = None
        self.new_coors = None

        assert self.cam_type in CamType.__members__, f"cam_type must be one of {CamType.__members__.keys()}"
        self.cam_type = CamType[self.cam_type]

        post_fix = "_sampled" if self.sampled else ""

        self.rgb_path = os.path.join(self.data_dir, self.scene_name, f'{self.scene_id:03d}'+post_fix, "rgb", self.cam_type.name)
        self.depth_path = os.path.join(self.data_dir, self.scene_name, f'{self.scene_id:03d}'+post_fix, "depth", self.cam_type.name)
        self.mask_path = os.path.join(self.data_dir, self.scene_name, f'{self.scene_id:03d}'+post_fix, "mask", self.cam_type.name)
        
        data_info_name = self.scene_name.split("_")[1] + f"_{self.scene_id:03d}"
        with open(os.path.join(self.data_dir, self.scene_name, f'{self.scene_id:03d}{post_fix}/{data_info_name}.json')) as f:
            self.cam_info = json.load(f)
            self.n_frames = len(self.cam_info['frames'])
        
        with open(os.path.join(self.data_dir, self.scene_name, f'{self.scene_id:03d}{post_fix}/sceneObject.json')) as f:
            self.instance_info = json.load(f)
    
    def rgbd2pcd(self, rgbs, depths, intrinsics, c2ws):
        # Assuming rgbs is of shape (N, 3, H, W), depths is of shape (N, 1, H, W), and c2ws is of shape (N, 4, 4)
        N, _, H, W = rgbs.shape
        if len(intrinsics.shape) == 2:
            intrinsics = intrinsics[None]
        intrinsics = torch.tensor(intrinsics, dtype=rgbs.dtype, device=rgbs.device)

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

            # This part distinguish different datasets
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
        frame_ids = frame_ids if frame_ids is not None else list(range(len(self.cam_info)))
        for i in tqdm(range(self.n_frames), desc="Loading Data"):
            if i in frame_ids:
                cam_id = int(self.cam_info['frames'][i]['id'])
                rgb = np.array(Image.open(os.path.join(self.rgb_path, f"{cam_id+1:06d}.jpg")))
                depth = np.array(Image.open(os.path.join(self.depth_path, f"{cam_id+1:06d}.png"))) / 10.0  # mm to cm
                mask = np.array(Image.open(os.path.join(self.mask_path, f"{cam_id+1:06d}.png")))[..., :3]

                cam_position = self.cam_info['frames'][i]['cameras'][self.cam_type.value-1]['position']
                cam_position = {k: float(v) for k, v in cam_position.items()}
                cam_rotation = self.cam_info['frames'][i]['cameras'][self.cam_type.value-1]['rotation']
                cam_rotation = {k: float(v) for k, v in cam_rotation.items()}
                c2w = np.eye(4)
                c2w[:3, :3] = ypr2rotmat(cam_rotation['y'], cam_rotation['p'], cam_rotation['r'])
                c2w[:3, 3] = np.array([cam_position['x'], cam_position['y'], cam_position['z']])

                rgbs.append(torch.tensor(rgb, dtype=self.dtype, device=self.device).permute(2, 0, 1))
                depths.append(torch.tensor(depth[None], dtype=self.dtype, device=self.device))
                masks.append(torch.tensor(mask, dtype=self.dtype, device=self.device).permute(2, 0, 1))
                c2ws.append(torch.tensor(c2w, dtype=self.dtype, device=self.device))
        
        self.n_frames = len(rgbs)
        rgbs = torch.stack(rgbs, dim=0) / 255.0
        depths = torch.stack(depths, dim=0)
        c2ws = torch.stack(c2ws, dim=0)
        N, _, H, W = rgbs.shape

        fov = float(self.cam_info['cameras'][self.cam_type.value-1]['fov'])
        f = W / (2 * np.tan(np.deg2rad(fov/2)))
        intrinsics = np.array([[f, 0, W/2], [0, f, H/2], [0, 0, 1]])

        p_world, rgb_world = self.rgbd2pcd(rgbs, depths, intrinsics, c2ws)  # Shape: (N, H*W, 3), (N, H*W, 3)
        p_world = process_frames(p_world.reshape(N, H, W, 3).permute(0, 3, 1, 2), self.h, self.w)  # Shape: (N, 3, h, w)
        rgb_world = process_frames(rgb_world.reshape(N, H, W, 3).permute(0, 3, 1, 2), self.h, self.w)  # Shape: (N, 3, h, w)
        flows, past_flows, mask_bwds = self.load_flow(frame_ids=frame_ids, future_flow=True, past_flow=True, gts=rgb_world)
        flow_ids = get_flowid(rgb_world, flows, mask_bwds, rgb_threshold=rgb_threshold)

        del rgbs, depths  # Free up memory

        self.unq_inv = voxelization(flow_ids.reshape(-1), 
                                    rgb_world.permute(0, 2, 3, 1).reshape(-1, 3), 
                                    p_world.permute(0, 2, 3, 1).reshape(-1, 3),
                                    self.voxel_size, contract=self.contract)

        return rgb_world, p_world, c2ws, flows, past_flows, mask_bwds
    
    

