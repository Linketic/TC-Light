import os
import json
import numpy as np
import torch
import cv2

from tqdm import tqdm
from .video_dataparser import VideoDataParser
from utils.flow_utils import get_flowid
from utils.general_utils import voxelization, process_frames

class CarlaDataParser(VideoDataParser):

    def __init__(self, data_config, device="cuda", dtype=torch.float32):
        self.data_dir = getattr(data_config, "data_dir", "data/carla/data_collect_town01_results")
        self.scene_name = getattr(data_config, "scene_name", "routes_town01_02_06_20_36_50")
        self.flow_model = getattr(data_config, "flow_model", "memflow")
        self.fov = getattr(data_config, "fov", 90)
        self.x_shift = getattr(data_config, "x_shift", 1.5)
        self.y_shift = getattr(data_config, "y_shift", 0.0)
        self.z_shift = getattr(data_config, "z_shift", 2.5)
        self.voxel_size = getattr(data_config, "voxel_size", None)
        self.apply_mask = getattr(data_config, "apply_mask", True)
        self.contract = getattr(data_config, "contract", False)
        self.fps = getattr(data_config, "fps", 30)
        self.alpha = getattr(data_config, "alpha", 0.1)
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
        N, _, H, W = rgbs.shape
        if len(intrinsics.shape) == 2:
            intrinsics = intrinsics[None]
        intrinsics = torch.tensor(intrinsics, dtype=torch.float32, device=rgbs.device)

        with torch.no_grad():
            pos_x, pos_y = torch.meshgrid(
                torch.arange(W, device=rgbs.device),
                torch.arange(H, device=rgbs.device),
                indexing='xy')

            pos_x = pos_x.unsqueeze(0).expand(N, -1, -1)
            pos_y = pos_y.unsqueeze(0).expand(N, -1, -1)
            p_img = torch.stack([pos_x, pos_y], dim=-1).reshape(N, -1, 2)

            x_cam = (p_img[:, :, 0] - intrinsics[:, 0, 2].unsqueeze(1)) * depths.reshape(N, -1) / intrinsics[:, 0, 0].unsqueeze(1)
            y_cam = (p_img[:, :, 1] - intrinsics[:, 1, 2].unsqueeze(1)) * depths.reshape(N, -1) / intrinsics[:, 1, 1].unsqueeze(1)

            p_cam_homo = torch.stack([
                x_cam, y_cam, depths.reshape(N, -1), torch.ones_like(x_cam, device=rgbs.device)
            ], dim=-1)

            p_cam_homo = p_cam_homo[:, :, [2, 0, 1, 3]]
            p_cam_homo[:, 1:3] *= -1
            p_world = torch.matmul(p_cam_homo, c2ws.transpose(-2, -1))[:, :, :3]
            rgb_world = rgbs.permute(0, 2, 3, 1).reshape(N, -1, 3)

        return p_world, rgb_world

    @torch.no_grad()
    def load_video(self, frame_ids=None):
        rgbs = []
        frame_ids = frame_ids if frame_ids is not None else list(range(self.n_frames))
        for i in tqdm(range(len(os.listdir(self.extrinsic_path))), desc="Loading Data"):
            if i in frame_ids:
                rgb = cv2.imread(os.path.join(self.rgb_path, f"{i:04d}.png"))
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                rgbs.append(torch.tensor(rgb, dtype=self.dtype).permute(2, 0, 1))

        self.n_frames = len(rgbs)
        rgbs = torch.stack(rgbs, dim=0) / 255.0
        rgbs = rgbs.to(self.device)
        rgbs = process_frames(rgbs, self.h, self.w)

        return rgbs

    @torch.no_grad()
    def load_data(self, frame_ids=None, rgb_threshold=0.01):
        rgbs, depths, masks, c2ws = [], [], [], []
        frame_ids = frame_ids if frame_ids is not None else list(range(self.n_frames))

        for i in tqdm(range(len(os.listdir(self.extrinsic_path))), desc="Loading Data"):
            if i in frame_ids:
                rgb = cv2.imread(os.path.join(self.rgb_path, f"{i:04d}.png"))
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                mask = cv2.imread(os.path.join(self.mask_path, f"{i:04d}.png"))
                depth = cv2.imread(os.path.join(self.depth_path, f"{i:04d}.png"))
                depth = (depth[:, :, 2] + depth[:, :, 1] * 256.0 + depth[:, :, 0] * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1)
                depth = depth * 1000.0

                with open(os.path.join(self.extrinsic_path, f"{i:04d}.json")) as f:
                    c2w = np.array(json.load(f))
                    c2w[0, 3] += self.x_shift
                    c2w[1, 3] += self.y_shift
                    c2w[2, 3] += self.z_shift

                rgbs.append(torch.tensor(rgb, dtype=self.dtype).permute(2, 0, 1))
                depths.append(torch.tensor(depth[None], dtype=self.dtype))
                masks.append(torch.tensor(mask, dtype=self.dtype).permute(2, 0, 1))
                c2ws.append(torch.tensor(c2w, dtype=self.dtype))

        self.n_frames = len(rgbs)
        rgbs = torch.stack(rgbs, dim=0) / 255.0
        depths = torch.stack(depths, dim=0)
        masks = torch.stack(masks, dim=0)
        c2ws = torch.stack(c2ws, dim=0)
        N, _, H, W = rgbs.shape

        f = W / (2 * np.tan(np.deg2rad(self.fov / 2)))
        intrinsics = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])

        p_world, rgb_world = self.rgbd2pcd(rgbs, depths, intrinsics, c2ws)
        del rgbs, depths

        rgb_world = process_frames(rgb_world.reshape(N, H, W, 3).permute(0, 3, 1, 2), self.h, self.w)
        flows, past_flows, mask_bwds = self.load_flow(frame_ids=frame_ids, future_flow=True, past_flow=True, gts=rgb_world)
        flow_ids = get_flowid(rgb_world, flows, mask_bwds, rgb_threshold=rgb_threshold).view(-1, 1)

        torch.cuda.empty_cache()

        rgb_world = rgb_world.permute(0, 2, 3, 1).reshape(-1, 3).to(self.device)
        p_world = process_frames(p_world.reshape(N, H, W, 3).permute(0, 3, 1, 2), self.h, self.w)
        p_world = p_world.permute(0, 2, 3, 1).reshape(-1, 3).to(self.device)

        if self.apply_mask:
            masks = process_frames(masks.reshape(N, H, W, 3).permute(0, 3, 1, 2), self.h, self.w)[:, 0:1]
            masks = masks.permute(0, 2, 3, 1).reshape(-1).to(self.device)
        else:
            masks = None

        self.unq_inv = voxelization(flow_ids, rgb_world, p_world,
                                    self.voxel_size, instance_ids=masks,
                                    contract=self.contract)

        torch.cuda.empty_cache()
        return rgb_world, p_world, c2ws, flows, past_flows, mask_bwds