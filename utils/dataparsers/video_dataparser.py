import os
import cv2
import torch

from tqdm import tqdm
from evaluation import eval_utils as eu

from plugin.VidToMe.utils import load_video as _load_video
from utils.general_utils import voxelization, process_frames
from utils.flow_utils import get_soft_mask_bwds, get_key_mask_bwds, get_flowid

class VideoDataParser:

    def __init__(self, 
                 data_config,
                 device="cuda",
                 dtype=torch.float32):

        self.rgb_path = data_config.rgb_path
        self.fps = 30 if not hasattr(data_config, "fps") else data_config.fps
        self.alpha = 0.5 if not hasattr(data_config, "alpha") else data_config.alpha
        self.beta = 1e2 if not hasattr(data_config, "beta") else data_config.beta
        self.flow_model = "memflow" if not hasattr(data_config, "flow_model") else data_config.flow_model
        self.h, self.w = data_config.height, data_config.width
        self.apply_opt = data_config.apply_opt
        self.voxel_size = None
        self.device = device
        self.dtype = dtype
        self.unq_inv = None

        if self.rgb_path.endswith(".mp4") or self.rgb_path.endswith(".gif") or self.rgb_path.endswith(".avi"):
            cap = cv2.VideoCapture(self.rgb_path)
            self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            self.n_frames = len([name for name in os.listdir(self.rgb_path) if os.path.isfile(os.path.join(self.rgb_path, name))])
    
    @torch.no_grad()
    def load_video(self, frame_ids=None, rgb_threshold=0.01):
        rgbs = _load_video(self.rgb_path, self.h, self.w, 
                           frame_ids=frame_ids, device=self.device, base=8)
        if rgbs.min() < 0:  # if normalized to [-1, 1]
            rgbs = (rgbs + 1.0) * 127.0 / 255.0
        frame_ids = frame_ids if frame_ids is not None else list(range(rgbs.shape[0]))

        self.n_frames = rgbs.shape[0]

        if self.apply_opt:
            future_flows, past_flows, mask_bwds, _, _, _ = self.load_flow(frame_ids=frame_ids, future_flow=True, past_flow=True, gts=rgbs)

            flow_ids = get_flowid(rgbs, future_flows, mask_bwds, rgb_threshold=rgb_threshold)

            self.unq_inv = voxelization(flow_ids.reshape(-1), 
                                        rgbs.permute(0, 2, 3, 1).reshape(-1, 3), 
                                        None, None)
        else:
            future_flows, past_flows = None, None
            mask_bwds = None

        return rgbs, None, None, future_flows, past_flows, mask_bwds
    
    @torch.no_grad()
    def load_flow(self, frame_ids=None, future_flow=False, past_flow=False, gts=None, target_ids=None, save_flow=True, diff_threshold=0.1):
        flows, past_flows, target_flows, src_flows = [], [], [], []

        if target_ids is not None:
            assert len(target_ids) == len(frame_ids), "target_ids and frame_ids should have the same length"

        if self.flow_model.lower() == 'raft':
            model = eu.prepare_raft_model(self.device)
        elif self.flow_model.lower() == 'memflow':
            model = eu.prepare_memflow_model(self.device)
            gts = gts * 2.0 - 1.0
            future_flow_prev, past_flow_prev, target_flow_prev, src_flow_prev = None, None, None, None
        else:
            raise NotImplementedError(f"{self.flow_model} is not implemented yet.")
        
        future_flow_path = self.create_folder(f"future_flow_{self.flow_model.lower()}")
        past_flow_path = self.create_folder(f"past_flow_{self.flow_model.lower()}")
        target_flow_path = self.create_folder(f"target_flow_{self.flow_model.lower()}")
        src_flow_path = self.create_folder(f"src_flow_{self.flow_model.lower()}")
        
        if save_flow:
            print(f"[INFO] Saving future flows to {future_flow_path} as .pt files")
            print(f"[INFO] Saving past flows to {past_flow_path} as .pt files")
            print(f"[INFO] Saving target flows to {target_flow_path} as .pt files")
            print(f"[INFO] Saving src flows to {src_flow_path} as .pt files")
        
        for idx in tqdm(range(len(gts)), desc="Loading Flows"):
            if future_flow:
                if os.path.exists(os.path.join(future_flow_path, "{:04d}.pt".format(frame_ids[idx]))):
                    flow_fwd = torch.load(os.path.join(future_flow_path, "{:04d}.pt".format(frame_ids[idx])))
                else:
                    flow_fwd, future_flow_prev = self.calc_flow(idx, gts.shape[0] - 1, gts[idx:idx+1], 
                                                                gts[idx+1:idx+2], model, future_flow_prev)
                    if save_flow:
                        torch.save(flow_fwd.cpu(), os.path.join(future_flow_path, "{:04d}.pt".format(frame_ids[idx])))

                flows.append(flow_fwd[0].cpu())

            if past_flow:
                if os.path.exists(os.path.join(past_flow_path, "{:04d}.pt".format(frame_ids[idx]))):
                    flow_bwd = torch.load(os.path.join(past_flow_path, "{:04d}.pt".format(frame_ids[idx])))
                else:
                    flow_bwd, past_flow_prev = self.calc_flow(idx, 0, gts[idx:idx+1],
                                                              gts[idx-1:idx], model, past_flow_prev)
                    if save_flow:
                        torch.save(flow_bwd.cpu(), os.path.join(past_flow_path, "{:04d}.pt".format(frame_ids[idx])))

                past_flows.append(flow_bwd[0].cpu())

            if target_ids is not None:
                target_id = target_ids[idx]

                if os.path.exists(os.path.join(target_flow_path, "{:04d}_{:04d}.pt".format(frame_ids[idx], frame_ids[target_id]))):
                    target_flow = torch.load(os.path.join(target_flow_path, "{:04d}_{:04d}.pt".format(frame_ids[idx], frame_ids[target_id])))
                    src_flow = torch.load(os.path.join(src_flow_path, "{:04d}_{:04d}.pt".format(frame_ids[target_id], frame_ids[idx])))
                else:
                    target_flow, _ = self.calc_flow(idx, target_id, gts[idx:idx+1], gts[target_id:target_id+1], model)
                    src_flow, _ = self.calc_flow(target_id, idx, gts[target_id:target_id+1], gts[idx:idx+1], model)

                    if save_flow:
                        torch.save(target_flow.cpu(), os.path.join(target_flow_path, "{:04d}_{:04d}.pt".format(frame_ids[idx], frame_ids[target_id])))
                        torch.save(src_flow.cpu(), os.path.join(src_flow_path, "{:04d}_{:04d}.pt".format(frame_ids[target_id], frame_ids[idx])))

                target_flows.append(target_flow[0].cpu())
                src_flows.append(src_flow[0].cpu())

        del model  # Free up memory
        
        flows = self.process_flow(flows) if future_flow else None
        past_flows = self.process_flow(past_flows) if past_flow else None
        target_flows = self.process_flow(target_flows) if target_ids is not None else None
        src_flows = self.process_flow(src_flows) if target_ids is not None else None

        mask_bwds_st = None if target_ids is None else get_key_mask_bwds(gts, target_ids, target_flows, src_flows, alpha=self.alpha, diff_threshold=diff_threshold)
        mask_bwds = None if not future_flow or not past_flow else get_soft_mask_bwds(gts, flows, past_flows, alpha=self.alpha, diff_threshold=diff_threshold)

        return flows, past_flows, mask_bwds, target_flows, src_flows, mask_bwds_st
    
    def create_folder(self, save_name):
        if os.path.isdir(self.rgb_path):
            save_path = os.path.join(self.rgb_path, save_name)
        else:
            ext = os.path.splitext(self.rgb_path)[-1]
            save_path = self.rgb_path.replace(ext, f"_{save_name}")
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        return save_path
    
    def process_flow(self, flow: list[torch.Tensor]):
        flow = torch.stack(flow, dim=0)
        N, _, H, W = flow.shape
        flow = process_frames(flow, self.h, self.w).to(self.device)
        scale_factor = max(self.w / W, self.h / H)
        flow *= scale_factor

        return flow

    def calc_flow(self, idx, zero_flow_idx, src_gt, tar_gt, model, flow_prev=None):
        if idx == zero_flow_idx:
            flow_fwd = torch.zeros_like(src_gt[:, :2])
        else:
            padder = eu.InputPadder(src_gt.shape)
            src_gt, tar_gt = padder.pad(src_gt, tar_gt)
            if self.flow_model.lower() == 'raft':
                _, flow_fwd = model(src_gt, tar_gt, iters=20, test_mode=True)
            else:
                input = torch.cat([src_gt, tar_gt], dim=0)
                flow_low, flow_pre = model.step(input[None], flow_init=flow_prev)
                flow_fwd = padder.unpad(flow_pre).cpu()
                flow_prev = eu.forward_interpolate(flow_low[0])[None].cuda()
        
        return flow_fwd, flow_prev