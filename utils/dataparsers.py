import os
import sys
import re
import uuid
import random
import imageio
import numpy as np
import torch
import torch_scatter
import torchvision.transforms as T

from scipy import misc
from PIL import Image
from tqdm import tqdm

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

def RGBD2PCD(rgbs, depths, intrinsics, c2ws):
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

        # Transform to blender coordinate system
        p_cam_homo[:, :, 1:3] *= -1

        # Transform to world coordinates
        p_world = torch.matmul(p_cam_homo, c2ws.transpose(-2, -1))[:, :, :3]  # Shape: (N, H*W, 3)

        # Reshape rgb to (N, H*W, 3)
        rgb_world = rgbs.permute(0, 2, 3, 1).reshape(N, -1, 3)  # Shape: (N, H*W, 3)
    
    return p_world, rgb_world

def process_frames(frames, h, w):

    fh, fw = frames.shape[-2:]
    scale_factor = max(w / fw, h / fh)
    nw = int(round(fw * scale_factor))
    nh = int(round(fh * scale_factor))
    size = (nh, nw)

    assert len(frames.shape) >= 3
    if len(frames.shape) == 3:
        frames = [frames]

    print(
        f"[INFO] frame size {(fh, fw)} resize to {size} and centercrop to {(h, w)}")

    frame_ls = []
    for frame in frames:
        resized_frame = T.Resize(size)(frame)
        cropped_frame = T.CenterCrop([h, w])(resized_frame)
        # croped_frame = T.FiveCrop([h, w])(resized_frame)[0]
        frame_ls.append(cropped_frame)
    return torch.stack(frame_ls)

def voxelization(p_world, feats, voxel_size, xyz_min=None):
    with torch.no_grad():
        # automatically determine the voxel size
        N, P, C = p_world.shape
        p_world = p_world.reshape(N*P, C)
        feats = feats.reshape(N*P, -1)
        if xyz_min is None:
            xyz_min = torch.min(p_world.reshape(-1, 3), dim=0).values
        voxel_index = torch.div(p_world - xyz_min[None, :], voxel_size[None, :], rounding_mode='floor')
        voxel_coords = voxel_index * voxel_size[None, :] + xyz_min[None, :] + voxel_size[None, :] / 2

        new_coors, unq_inv, unq_cnt = torch.unique(voxel_coords, return_inverse=True, return_counts=True, dim=0)
        print("[INFO] Number of unique voxels: ", new_coors.shape[0])
        # feat_mean = torch_scatter.scatter(feats, unq_inv, dim=0, reduce='mean')

        # new_feats = feat_mean[unq_inv]

        return unq_inv

class SceneFlowDataParser:

    def __init__(self, 
                 data_config,
                 device="cuda"):

        self.data_dir = "data/sceneflow" if not hasattr(data_config, "data_dir") else data_config.data_dir
        self.scene_path = "15mm_focallength/scene_backwards/fast" if not hasattr(data_config, "scene_path") else data_config.scene_path
        self.stereo_sel = "left" if not hasattr(data_config, "stereo_sel") else data_config.stereo_sel
        self.voxel_size = 0.1 if not hasattr(data_config, "voxel_size") else data_config.voxel_size
        self.h, self.w = data_config.height, data_config.width
        self.device = device
        self.unq_inv = None
        self.flows = None

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
    
    def load_video(self, frame_ids=None):
        rgbs, depths, c2ws = [], [], []
        for i in tqdm(range(len(self.cam_info)), desc="Loading Data"):
            if i in frame_ids:
                rgb = read(os.path.join(self.rgb_path, "{:04d}.png".format(self.cam_info[i]["frame_id"])))
                disparity = read(os.path.join(self.disparity_path, "{:04d}.pfm".format(self.cam_info[i]["frame_id"])))
                depth = (self.intrinsics[0, 0]* 1.0 / disparity)
                c2w = self.cam_info[i][self.stereo_sel]

                rgbs.append(torch.tensor(rgb, dtype=torch.float32, device=self.device).permute(2, 0, 1))
                depths.append(torch.tensor(depth[None], dtype=torch.float32, device=self.device))
                c2ws.append(torch.tensor(c2w, dtype=torch.float32, device=self.device))
        
        rgbs = torch.stack(rgbs, dim=0) / 255.0
        depths = torch.stack(depths, dim=0)
        c2ws = torch.stack(c2ws, dim=0)
        N, _, H, W = rgbs.shape

        # Assuming rgb is of shape (N, 3, H, W), depth is of shape (N, 1, H, W), and c2w is of shape (N, 4, 4)
        p_world, rgb_world = RGBD2PCD(rgbs, depths, self.intrinsics, c2ws)  # Shape: (N, H*W, 3), (N, H*W, 3)
        p_world = process_frames(p_world.reshape(N, H, W, 3).permute(0, 3, 1, 2), self.h, self.w)  # Shape: (N, 3, h, w)
        rgb_world = process_frames(rgb_world.reshape(N, H, W, 3).permute(0, 3, 1, 2), self.h, self.w)  # Shape: (N, 3, h, w)
        del rgbs, depths, c2ws  # Free up memory

        # Get mapping from frame to 3D prior
        voxel_size = torch.tensor([self.voxel_size] * 3, dtype=p_world.dtype, device=p_world.device)
        self.unq_inv = voxelization(p_world.permute(0, 2, 3, 1).reshape(N, -1, 3), 
                                    rgb_world.permute(0, 2, 3, 1).reshape(N, -1, 3), voxel_size)

        return rgb_world
    
    def load_video_flow(self, frame_ids=None, past_flow=False):
        rgbs, flows, past_flows = [], [], []
        for i in tqdm(range(len(self.cam_info)), desc="Loading Data"):
            if i in frame_ids:
                rgb = read(os.path.join(self.rgb_path, "{:04d}.png".format(self.cam_info[i]["frame_id"])))
                optical_flow_future = read(os.path.join(self.future_flow_path, "OpticalFlowIntoFuture_{:04d}_L.pfm".format(self.cam_info[i]["frame_id"])))

                rgbs.append(torch.tensor(rgb, dtype=torch.float32, device=self.device).permute(2, 0, 1))
                flows.append(torch.tensor(optical_flow_future.copy(), dtype=torch.float32, device=self.device).permute(2, 0, 1))

                if past_flow:
                    optical_flow_past = read(os.path.join(self.past_flow_path, "OpticalFlowIntoPast_{:04d}_L.pfm".format(self.cam_info[i]["frame_id"])))
                    past_flows.append(torch.tensor(optical_flow_past.copy(), dtype=torch.float32, device=self.device).permute(2, 0, 1))
            
        rgbs = torch.stack(rgbs, dim=0) / 255.0
        flows = torch.stack(flows, dim=0)
        
        N, _, H, W = rgbs.shape
        flows = process_frames(flows, self.h, self.w)  # Shape: (N, 3, h, w)
        rgbs = process_frames(rgbs, self.h, self.w)  # Shape: (N, 3, h, w)

        scale_factor = max(self.w / W, self.h / H)
        flows *= scale_factor

        self.flows = flows

        if past_flow:
            past_flows = torch.stack(past_flows, dim=0)
            past_flows = process_frames(past_flows, self.h, self.w)
            past_flows *= scale_factor
            self.past_flows = past_flows
        else:
            self.past_flows = None

        return rgbs



