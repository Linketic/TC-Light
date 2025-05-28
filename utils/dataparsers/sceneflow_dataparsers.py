import os
import sys
import re
import imageio
import numpy as np
import torch
import torchvision.transforms as T

from scipy import misc
from tqdm import tqdm
from evaluation import eval_utils as eu

from utils.general_utils import voxelization, process_frames
from utils.flow_utils import get_soft_mask_bwds, get_flowid

def read(file):
    if file.endswith('.float3'): return readFloat(file)
    elif file.endswith('.flo'): return readFlow(file)
    elif file.endswith('.ppm'): return readImage(file)
    elif file.endswith('.pgm'): return readImage(file)
    elif file.endswith('.png'): return readImage(file)
    elif file.endswith('.jpg'): return readImage(file)
    elif file.endswith('.pfm'): return readPFM(file)[0]
    else: raise Exception('don\'t know how to read %s' % file)

def write(file, data):
    if file.endswith('.float3'): return writeFloat(file, data)
    elif file.endswith('.flo'): return writeFlow(file, data)
    elif file.endswith('.ppm'): return writeImage(file, data)
    elif file.endswith('.pgm'): return writeImage(file, data)
    elif file.endswith('.png'): return writeImage(file, data)
    elif file.endswith('.jpg'): return writeImage(file, data)
    elif file.endswith('.pfm'): return writePFM(file, data)
    else: raise Exception('don\'t know how to write %s' % file)

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

def writePFM(file, image, scale=1):
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n'.encode() % scale)

    image.tofile(file)

def readFlow(name):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        return readPFM(name)[0][:,:,0:2]

    f = open(name, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

    return flow.astype(np.float32)

def readImage(name):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        data = readPFM(name)[0]
        if len(data.shape)==3:
            return data[:,:,0:3]
        else:
            return data

    return imageio.imread(name)

def writeImage(name, data):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        return writePFM(name, data, 1)

    return misc.imsave(name, data)

def writeFlow(name, flow):
    f = open(name, 'wb')
    f.write('PIEH'.encode('utf-8'))
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    flow = flow.astype(np.float32)
    flow.tofile(f)

def readFloat(name):
    f = open(name, 'rb')

    if(f.readline().decode("utf-8"))  != 'float\n':
        raise Exception('float file %s did not contain <float> keyword' % name)

    dim = int(f.readline())

    dims = []
    count = 1
    for i in range(0, dim):
        d = int(f.readline())
        dims.append(d)
        count *= d

    dims = list(reversed(dims))

    data = np.fromfile(f, np.float32, count).reshape(dims)
    if dim > 2:
        data = np.transpose(data, (2, 1, 0))
        data = np.transpose(data, (1, 0, 2))

    return data

def writeFloat(name, data):
    f = open(name, 'wb')

    dim=len(data.shape)
    if dim>3:
        raise Exception('bad float file dimension: %d' % dim)

    f.write(('float\n').encode('ascii'))
    f.write(('%d\n' % dim).encode('ascii'))

    if dim == 1:
        f.write(('%d\n' % data.shape[0]).encode('ascii'))
    else:
        f.write(('%d\n' % data.shape[1]).encode('ascii'))
        f.write(('%d\n' % data.shape[0]).encode('ascii'))
        for i in range(2, dim):
            f.write(('%d\n' % data.shape[i]).encode('ascii'))

    data = data.astype(np.float32)
    if dim==2:
        data.tofile(f)

    else:
        np.transpose(data, (2, 0, 1)).tofile(f)

def readCamInfo(file):
    # read camera info txt file, the structure is:
    # ...	
    # Frame <frame_id>\n	frame_id is the frame index. All images and data files for this frame carry this name, as a four-digit number with leading zeroes for padding.
    # L T00 T01 T02 T03 T10 ... T33\n	Camera-to-world 4x4 matrix for the left view of the stereo pair in row-major order, i.e. (T00 T01 T02 T03) encodes the uppermost row from left to right.
    # R T00 T01 T02 T03 T10 ... T33\n	Ditto for the right view of the stereo pair.
    # \n	(an empty line)
    # Frame <frame_id>\n	(the next frame's index)
    # ...	(and so on)

    cam_info = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for i in tqdm(range(0, len(lines), 4), desc="Reading camera data"):
            frame_id = int(lines[i].split()[1])
            T_left = np.array(lines[i+1].split()[1:], dtype=float).reshape(4, 4)
            T_right = np.array(lines[i+2].split()[1:], dtype=float).reshape(4, 4)
            cam_info.append({
                "frame_id": frame_id,
                "left": T_left,
                "right": T_right
            })
            
    return cam_info

class SceneFlowDataParser:

    def __init__(self, 
                 data_config,
                 device="cuda",
                 dtype=torch.float32):

        self.data_dir = "data/sceneflow" if not hasattr(data_config, "data_dir") else data_config.data_dir
        self.scene_path = "15mm_focallength/scene_backwards/fast" if not hasattr(data_config, "scene_path") else data_config.scene_path
        self.stereo_sel = "left" if not hasattr(data_config, "stereo_sel") else data_config.stereo_sel
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

        assert self.stereo_sel in ['left', 'right'], "stereo_sel must be either 'left' or 'right'"
        assert self.scene_path.split('/')[0] in ['15mm_focallength', '35mm_focallength'], "scene_path must be either '15mm_focallength' or '35mm_focallength'"
        assert self.scene_path.split('/')[1] in ['scene_backwards', 'scene_forwards'], "scene_path must be either 'scene_backwards' or 'scene_forwards'"
        assert self.scene_path.split('/')[2] in ['fast', 'slow'], "scene_path must be either 'fast' or 'slow'"

        self.rgb_path = os.path.join(self.data_dir, "frames_cleanpass", self.scene_path, self.stereo_sel)
        self.disparity_path = os.path.join(self.data_dir, "disparity", self.scene_path, self.stereo_sel)
        self.future_flow_path = os.path.join(self.data_dir, "optical_flow", self.scene_path, "into_future", self.stereo_sel)
        self.past_flow_path = os.path.join(self.data_dir, "optical_flow", self.scene_path, "into_past", self.stereo_sel)

        if "15mm" in self.scene_path:
            self.intrinsics = np.array([[450.0, 0.0, 479.5], [0.0, 450.0, 269.5], [0.0, 0.0, 1.0]])
        else:
            self.intrinsics = np.array([[1050.0, 0.0, 479.5], [0.0, 1050.0, 269.5], [0.0, 0.0, 1.0]])
        
        self.cam_info = readCamInfo(os.path.join(self.data_dir, "camera_data", self.scene_path, "camera_data.txt"))
        self.n_frames = len(self.cam_info)
    
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
            p_cam_homo[:, :, 1:3] *= -1

            # Transform to world coordinates
            p_world = torch.matmul(p_cam_homo, c2ws.transpose(-2, -1))[:, :, :3]  # Shape: (N, H*W, 3)

            # Reshape rgb to (N, H*W, 3)
            rgb_world = rgbs.permute(0, 2, 3, 1).reshape(N, -1, 3)  # Shape: (N, H*W, 3)
        
        return p_world, rgb_world
    
    @torch.no_grad()
    def load_video(self, frame_ids=None):
        rgbs = []
        frame_ids = frame_ids if frame_ids is not None else list(range(len(self.cam_info)))
        for i in tqdm(range(len(self.cam_info)), desc="Loading Data"):
            if i in frame_ids:
                rgb = read(os.path.join(self.rgb_path, "{:04d}.png".format(self.cam_info[i]["frame_id"])))
                rgbs.append(torch.tensor(rgb, dtype=self.dtype, device=self.device).permute(2, 0, 1))
        
        self.n_frames = len(rgbs)
        rgbs = torch.stack(rgbs, dim=0) / 255.0
        rgbs = process_frames(rgbs, self.h, self.w)  # Shape: (N, 3, h, w)

        return rgbs

    @torch.no_grad()
    def load_data(self, frame_ids=None, contract=False, rgb_threshold=0.01):
        rgbs, depths, c2ws = [], [], []
        frame_ids = frame_ids if frame_ids is not None else list(range(len(self.cam_info)))
        for i in tqdm(range(len(self.cam_info)), desc="Loading Data"):
            if i in frame_ids:
                rgb = read(os.path.join(self.rgb_path, "{:04d}.png".format(self.cam_info[i]["frame_id"])))
                disparity = read(os.path.join(self.disparity_path, "{:04d}.pfm".format(self.cam_info[i]["frame_id"])))
                depth = (self.intrinsics[0, 0] * 1.0 / disparity)
                c2w = self.cam_info[i][self.stereo_sel]

                rgbs.append(torch.tensor(rgb, dtype=self.dtype, device=self.device).permute(2, 0, 1))
                depths.append(torch.tensor(depth[None], dtype=self.dtype, device=self.device))
                c2ws.append(torch.tensor(c2w, dtype=self.dtype, device=self.device))
        
        self.n_frames = len(rgbs)
        rgbs = torch.stack(rgbs, dim=0) / 255.0
        depths = torch.stack(depths, dim=0)
        c2ws = torch.stack(c2ws, dim=0)
        N, _, H, W = rgbs.shape
        p_world, rgb_world = self.rgbd2pcd(rgbs, depths, self.intrinsics, c2ws)  # Shape: (N, H*W, 3), (N, H*W, 3)
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
    
    @torch.no_grad()
    def load_flow(self, frame_ids=None, future_flow=False, past_flow=False, gts=None, diff_threshold=0.1):
        flows, past_flows = [], []
        stereo_tag = "L" if self.stereo_sel == "left" else "R"
        raft_model = eu.prepare_raft_model(self.device) if self.use_raft else None
        for i in tqdm(range(len(self.cam_info)), desc="Loading Flows"):
            if i in frame_ids:
                if not self.use_raft:
                    if future_flow:
                        optical_flow_future = read(os.path.join(self.future_flow_path, "OpticalFlowIntoFuture_{:04d}_{}.pfm".format(self.cam_info[i]["frame_id"], stereo_tag)))
                        flows.append(torch.tensor(optical_flow_future.copy(), dtype=self.dtype, device=self.device).permute(2, 0, 1))

                    if past_flow:
                        optical_flow_past = read(os.path.join(self.past_flow_path, "OpticalFlowIntoPast_{:04d}_{}.pfm".format(self.cam_info[i]["frame_id"], stereo_tag)))
                        past_flows.append(torch.tensor(optical_flow_past.copy(), dtype=self.dtype, device=self.device).permute(2, 0, 1))
                else:
                    idx = torch.where(torch.tensor(frame_ids) == i)[0].item()
                    if future_flow:
                        if idx == gts.shape[0] - 1:
                            flow_fwd = torch.zeros_like(gts[0:1, :2])
                        else:
                            padder = eu.InputPadder(gts[idx:idx+1].shape)
                            gt, gt_next = padder.pad(gts[idx:idx+1], gts[idx+1:idx+2])
                            _, flow_fwd = raft_model(gt, gt_next, iters=20, test_mode=True)
                        flows.append(flow_fwd[0].cpu())

                    if past_flow:
                        if idx == 0:
                            flow_bwd = torch.zeros_like(gts[0:1, :2])
                        else:
                            padder = eu.InputPadder(gts[idx:idx+1].shape)
                            gt, gt_prev = padder.pad(gts[idx:idx+1], gts[idx-1:idx])
                            _, flow_bwd = raft_model(gt, gt_prev, iters=20, test_mode=True)
                        past_flows.append(flow_bwd[0].cpu())
        
        del raft_model  # Free up memory

        if future_flow:
            flows = torch.stack(flows, dim=0)
            N, _, H, W = flows.shape
            flows = process_frames(flows, self.h, self.w).to(self.device)
            scale_factor = max(self.w / W, self.h / H)
            flows *= scale_factor
        else:
            flows = None

        if past_flow:
            past_flows = torch.stack(past_flows, dim=0)
            N, _, H, W = past_flows.shape
            past_flows = process_frames(past_flows, self.h, self.w).to(self.device)
            scale_factor = max(self.w / W, self.h / H)
            past_flows *= scale_factor
        else:
            past_flows = None

        if future_flow and past_flow:
            mask_bwds = get_soft_mask_bwds(gts, flows, past_flows, alpha=self.alpha, diff_threshold=diff_threshold)
        else:
            mask_bwds = None

        return flows, past_flows, mask_bwds

