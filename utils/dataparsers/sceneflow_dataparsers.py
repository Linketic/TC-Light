import os
import sys
import re
import imageio
import numpy as np
import torch

from scipy import misc
from tqdm import tqdm
from utils.evaluation import eval_utils as eu

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

    def __init__(self, data_config, device="cuda", dtype=torch.float32):
        # Default config with fallback
        self.data_dir = getattr(data_config, "data_dir", "data/sceneflow")
        self.scene_path = getattr(data_config, "scene_path", "15mm_focallength/scene_backwards/fast")
        self.stereo_sel = getattr(data_config, "stereo_sel", "left")
        self.voxel_size = getattr(data_config, "voxel_size", None)
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

        # Validations
        stereo_options = ['left', 'right']
        assert self.stereo_sel in stereo_options, f"stereo_sel must be one of {stereo_options}"

        sp = self.scene_path.split('/')
        assert sp[0] in ['15mm_focallength', '35mm_focallength'], "Invalid focal length"
        assert sp[1] in ['scene_backwards', 'scene_forwards'], "Invalid scene direction"
        assert sp[2] in ['fast', 'slow'], "Invalid speed"

        # Paths
        self.rgb_path = os.path.join(self.data_dir, "frames_cleanpass", self.scene_path, self.stereo_sel)
        self.disparity_path = os.path.join(self.data_dir, "disparity", self.scene_path, self.stereo_sel)
        self.future_flow_path = os.path.join(self.data_dir, "optical_flow", self.scene_path, "into_future", self.stereo_sel)
        self.past_flow_path = os.path.join(self.data_dir, "optical_flow", self.scene_path, "into_past", self.stereo_sel)

        # Camera intrinsics
        self.intrinsics = np.array([[450.0, 0.0, 479.5], [0.0, 450.0, 269.5], [0.0, 0.0, 1.0]]) if "15mm" in self.scene_path \
                           else np.array([[1050.0, 0.0, 479.5], [0.0, 1050.0, 269.5], [0.0, 0.0, 1.0]])

        self.cam_info = readCamInfo(os.path.join(self.data_dir, "camera_data", self.scene_path, "camera_data.txt"))
        self.n_frames = len(self.cam_info)

    def rgbd2pcd(self, rgbs, depths, intrinsics, c2ws):
        N, _, H, W = rgbs.shape
        intrinsics = torch.tensor(intrinsics[None] if intrinsics.ndim == 2 else intrinsics,
                                  dtype=rgbs.dtype, device=rgbs.device)

        pos_x, pos_y = torch.meshgrid(torch.arange(W, device=rgbs.device),
                                      torch.arange(H, device=rgbs.device), indexing='xy')
        pos_x, pos_y = pos_x.expand(N, -1, -1), pos_y.expand(N, -1, -1)
        p_img = torch.stack([pos_x, pos_y], dim=-1).reshape(N, -1, 2)

        x_cam = (p_img[..., 0] - intrinsics[:, 0, 2].unsqueeze(1)) * depths.reshape(N, -1) / intrinsics[:, 0, 0].unsqueeze(1)
        y_cam = (p_img[..., 1] - intrinsics[:, 1, 2].unsqueeze(1)) * depths.reshape(N, -1) / intrinsics[:, 1, 1].unsqueeze(1)

        p_cam_homo = torch.stack([x_cam, -y_cam, -depths.reshape(N, -1), torch.ones_like(x_cam)], dim=-1)
        p_world = torch.matmul(p_cam_homo, c2ws.transpose(-2, -1))[:, :, :3]

        rgb_world = rgbs.permute(0, 2, 3, 1).reshape(N, -1, 3)
        return p_world, rgb_world

    @torch.no_grad()
    def load_video(self, frame_ids=None):
        rgbs = []
        frame_ids = frame_ids or list(range(self.n_frames))

        for i in tqdm(frame_ids, desc="Loading Data"):
            rgb = read(os.path.join(self.rgb_path, f"{self.cam_info[i]['frame_id']:04d}.png"))
            rgbs.append(torch.tensor(rgb, dtype=self.dtype, device=self.device).permute(2, 0, 1))

        rgbs = torch.stack(rgbs) / 255.0
        return process_frames(rgbs, self.h, self.w)

    @torch.no_grad()
    def load_data(self, frame_ids=None, contract=False, rgb_threshold=0.01):
        rgbs, depths, c2ws = [], [], []
        frame_ids = frame_ids or list(range(self.n_frames))

        for i in tqdm(frame_ids, desc="Loading Data"):
            fid = self.cam_info[i]['frame_id']
            rgb = read(os.path.join(self.rgb_path, f"{fid:04d}.png"))
            disp = read(os.path.join(self.disparity_path, f"{fid:04d}.pfm"))
            depth = self.intrinsics[0, 0] / disp
            c2w = self.cam_info[i][self.stereo_sel]

            rgbs.append(torch.tensor(rgb, dtype=self.dtype, device=self.device).permute(2, 0, 1))
            depths.append(torch.tensor(depth[None], dtype=self.dtype, device=self.device))
            c2ws.append(torch.tensor(c2w, dtype=self.dtype, device=self.device))

        rgbs, depths, c2ws = map(torch.stack, (rgbs, depths, c2ws))
        rgbs /= 255.0

        N, _, H, W = rgbs.shape
        p_world, rgb_world = self.rgbd2pcd(rgbs, depths, self.intrinsics, c2ws)

        p_world = process_frames(p_world.reshape(N, H, W, 3).permute(0, 3, 1, 2), self.h, self.w)
        rgb_world = process_frames(rgb_world.reshape(N, H, W, 3).permute(0, 3, 1, 2), self.h, self.w)

        flows, past_flows, mask_bwds = self.load_flow(frame_ids=frame_ids, future_flow=True, past_flow=True, gts=rgb_world)
        flow_ids = get_flowid(rgb_world, flows, mask_bwds, rgb_threshold)

        self.unq_inv = voxelization(flow_ids.reshape(-1, 1),
                                    rgb_world.permute(0, 2, 3, 1).reshape(-1, 3),
                                    p_world.permute(0, 2, 3, 1).reshape(-1, 3),
                                    self.voxel_size, contract=self.contract)

        return rgb_world, p_world, c2ws, flows, past_flows, mask_bwds

    @torch.no_grad()
    def load_flow(self, frame_ids=None, future_flow=False, past_flow=False, gts=None, diff_threshold=0.1):
        flows, past_flows = [], []
        stereo_tag = "L" if self.stereo_sel == "left" else "R"
        raft_model = eu.prepare_raft_model(self.device) if self.use_raft else None

        for i in tqdm(frame_ids, desc="Loading Flows"):
            idx = frame_ids.index(i)
            if self.use_raft:
                if future_flow:
                    if idx == len(gts) - 1:
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
            else:
                fid = self.cam_info[i]['frame_id']
                if future_flow:
                    path = os.path.join(self.future_flow_path, f"OpticalFlowIntoFuture_{fid:04d}_{stereo_tag}.pfm")
                    flow = read(path)
                    flows.append(torch.tensor(flow.copy(), dtype=self.dtype, device=self.device).permute(2, 0, 1))
                if past_flow:
                    path = os.path.join(self.past_flow_path, f"OpticalFlowIntoPast_{fid:04d}_{stereo_tag}.pfm")
                    flow = read(path)
                    past_flows.append(torch.tensor(flow.copy(), dtype=self.dtype, device=self.device).permute(2, 0, 1))

        del raft_model

        flows = process_frames(torch.stack(flows), self.h, self.w).to(self.device) if future_flow else None
        past_flows = process_frames(torch.stack(past_flows), self.h, self.w).to(self.device) if past_flow else None

        scale = max(self.w / flows.shape[-1], self.h / flows.shape[-2]) if future_flow else 1
        if flows is not None: flows *= scale
        if past_flows is not None: past_flows *= scale

        mask_bwds = get_soft_mask_bwds(gts, flows, past_flows, alpha=self.alpha, diff_threshold=diff_threshold) if future_flow and past_flow else None
        return flows, past_flows, mask_bwds
