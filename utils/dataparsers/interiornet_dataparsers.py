import os
import numpy as np
import torch
from tqdm import tqdm

from .video_dataparser import VideoDataParser
from .sceneflow_dataparsers import read
from utils.general_utils import voxelization, process_frames
from utils.flow_utils import get_flowid


def read_camera_intrinsic(file):
    """Reads camera intrinsics from a text file."""
    with open(file, 'r') as f:
        lines = f.readlines()

    resolution = list(map(int, lines[1].strip().split()))
    fx, fy = map(float, lines[3].strip().split())
    cx, cy = map(float, lines[5].strip().split())

    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])


def quaternion_to_rotation_matrix(q):
    """Converts quaternion to rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y**2 + z**2),     2 * (x*y - z*w),     2 * (x*z + y*w)],
        [    2 * (x*y + z*w), 1 - 2 * (x**2 + z**2),     2 * (y*z - x*w)],
        [    2 * (x*z - y*w),     2 * (y*z + x*w), 1 - 2 * (x**2 + y**2)]
    ])


def parse_visim_file(file_path):
    """Parses visim pose file into a timestamp->extrinsic matrix dict."""
    extrinsics_dict = {}

    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            data = line.strip().split(",")
            timestamp = int(data[0])
            position = np.array(list(map(float, data[1:4])))
            quaternion = np.array(list(map(float, data[4:8])))

            rot_mat = quaternion_to_rotation_matrix(quaternion)
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = rot_mat
            extrinsic[:3, 3] = position

            extrinsics_dict[timestamp] = extrinsic

    return extrinsics_dict


def normalize(v):
    return v / np.linalg.norm(v)


class InteriorNetDataParser(VideoDataParser):
    """InteriorNet dataset parser."""

    def __init__(self, data_config, device="cuda", dtype=torch.float32):
        self.data_dir = getattr(data_config, "data_dir", "data/interiornet/HD1/3FO4K35GPEA7")
        self.traj = getattr(data_config, "traj", 7)
        self.flow_model = getattr(data_config, "flow_model", "memflow")
        self.voxel_size = getattr(data_config, "voxel_size", None)
        self.apply_mask = getattr(data_config, "apply_mask", True)
        self.contract = getattr(data_config, "contract", False)
        self.use_raft = getattr(data_config, "use_raft", False)
        self.fps = getattr(data_config, "fps", 30)
        self.alpha = getattr(data_config, "alpha", 0.1)
        self.h = data_config.height
        self.w = data_config.width
        self.device = device
        self.dtype = dtype

        self.unq_inv = None
        self.new_coors = None

        traj_prefix = f"original_{self.traj}_{self.traj}"
        cam_info_prefix = f"velocity_angular_{self.traj}_{self.traj}"

        self.rgb_path = os.path.join(self.data_dir, traj_prefix, "cam0", "data")
        self.depth_path = os.path.join(self.data_dir, traj_prefix, "depth0", "data")
        self.mask_path = os.path.join(self.data_dir, traj_prefix, "label0", "data")

        self.intrinsics = read_camera_intrinsic(os.path.join(self.data_dir, cam_info_prefix, "cam0.info"))
        self.extrinsics_dict = parse_visim_file(os.path.join(self.data_dir, cam_info_prefix, "cam0_gt.visim"))
        self.timestamps = list(self.extrinsics_dict.keys())
        self.n_frames = len(self.timestamps)

    def rgbd2pcd(self, rgbs, depths, intrinsics, c2ws):
        """Converts RGB-D frames and camera transforms to point clouds."""
        N, _, H, W = rgbs.shape

        if intrinsics.ndim == 2:
            intrinsics = intrinsics[None]
        intrinsics = torch.tensor(intrinsics, dtype=torch.float32, device=rgbs.device)

        with torch.no_grad():
            pos_x, pos_y = torch.meshgrid(
                torch.arange(W, device=rgbs.device),
                torch.arange(H, device=rgbs.device),
                indexing="xy"
            )
            pos = torch.stack([pos_x, pos_y], dim=-1).expand(N, -1, -1, -1).reshape(N, -1, 2)

            x = (pos[..., 0] - intrinsics[:, 0, 2].unsqueeze(1)) * depths.reshape(N, -1) / intrinsics[:, 0, 0].unsqueeze(1)
            y = (pos[..., 1] - intrinsics[:, 1, 2].unsqueeze(1)) * depths.reshape(N, -1) / intrinsics[:, 1, 1].unsqueeze(1)

            p_cam = torch.stack([x, -y, -depths.reshape(N, -1), torch.ones_like(x)], dim=-1)
            p_world = torch.matmul(p_cam, c2ws.transpose(-2, -1))[:, :, :3]

            rgb = rgbs.permute(0, 2, 3, 1).reshape(N, -1, 3)

        return p_world, rgb

    @torch.no_grad()
    def load_video(self, frame_ids=None, rgb_threshold=0.01):
        """Loads RGB video frames as tensor."""
        frame_ids = frame_ids or list(range(len(self.timestamps)))
        rgbs = []

        for i in tqdm(frame_ids, desc="Loading RGB"):
            try:
                rgb = read(os.path.join(self.rgb_path, f"{self.timestamps[i]:019d}.png"))
                rgbs.append(torch.tensor(rgb, dtype=self.dtype).permute(2, 0, 1))
            except FileNotFoundError:
                print(f"Frame {self.timestamps[i]} not found, skipping.")

        self.n_frames = len(rgbs)
        rgbs = torch.stack(rgbs).to(self.device) / 255.0
        return process_frames(rgbs, self.h, self.w)

    @torch.no_grad()
    def load_data(self, frame_ids=None, rgb_threshold=0.01):
        """Loads RGB-D + mask + camera + flow information and performs voxelization."""
        frame_ids = frame_ids or list(range(len(self.timestamps)))
        rgbs, depths, masks, c2ws = [], [], [], []

        for i in tqdm(frame_ids, desc="Loading RGBD + Mask"):
            try:
                timestamp = self.timestamps[i]
                rgb = read(os.path.join(self.rgb_path, f"{timestamp:019d}.png"))
                mask = read(os.path.join(self.mask_path, f"{timestamp:019d}_instance.png"))
                depth = read(os.path.join(self.depth_path, f"{timestamp:019d}.png"))

                vs = np.arange(depth.shape[1])
                us = np.arange(depth.shape[0])
                vs = (vs - self.intrinsics[0, 2]) / self.intrinsics[0, 0]
                us = (us - self.intrinsics[1, 2]) / self.intrinsics[1, 1]
                depth = np.sqrt((depth / 1000.0)**2 / (1 + vs[np.newaxis, :]**2 + us[:, np.newaxis]**2))

                c2w = self.extrinsics_dict[timestamp]

                rgbs.append(torch.tensor(rgb, dtype=self.dtype).permute(2, 0, 1))
                depths.append(torch.tensor(depth[None], dtype=self.dtype))
                masks.append(torch.tensor(mask[None], dtype=self.dtype))
                c2ws.append(torch.tensor(c2w, dtype=self.dtype))

            except FileNotFoundError:
                print(f"Frame {self.timestamps[i]} not found, skipping.")

        rgbs = torch.stack(rgbs) / 255.0
        depths = torch.stack(depths)
        masks = torch.stack(masks)
        c2ws = torch.stack(c2ws)

        N, _, H, W = rgbs.shape

        p_world, rgb_world = self.rgbd2pcd(rgbs, depths, self.intrinsics, c2ws)
        rgb_world = process_frames(rgb_world.reshape(N, H, W, 3).permute(0, 3, 1, 2), self.h, self.w)

        flows, past_flows, mask_bwds = self.load_flow(
            frame_ids=frame_ids, future_flow=True, past_flow=True, gts=rgb_world)

        flow_ids = get_flowid(rgb_world, flows, mask_bwds, rgb_threshold=rgb_threshold).view(-1, 1)

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
