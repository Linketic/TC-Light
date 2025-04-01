import os
import cv2
import torch

from tqdm import tqdm
from evaluation import eval_utils as eu

from plugin.VidToMe.utils import load_video as _load_video
from utils.general_utils import voxelization, process_frames
from utils.flow_utils import get_mask_bwds, get_flowid

class VideoDataParser:

    def __init__(self, 
                 data_config,
                 device="cuda",
                 dtype=torch.float32):

        self.rgb_path = data_config.rgb_path
        self.fps = 30 if not hasattr(data_config, "fps") else data_config.fps
        self.alpha = 0.5 if not hasattr(data_config, "alpha") else data_config.alpha
        self.flow_model = "memflow" if not hasattr(data_config, "flow_model") else data_config.memflow
        self.h, self.w = data_config.height, data_config.width
        self.voxel_size = None
        self.device = device
        self.dtype = dtype
        self.unq_inv = None

        if self.rgb_path.endswith(".mp4") or self.rgb_path.endswith(".gif"):
            cap = cv2.VideoCapture(self.rgb_path)
            self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            self.n_frames = len(os.listdir(self.rgb_path))
    
    @torch.no_grad()
    def load_video(self, frame_ids=None, rgb_threshold=0.01):
        rgbs = _load_video(self.rgb_path, self.h, self.w, 
                           frame_ids=frame_ids, device=self.device, base=8)
        frame_ids = frame_ids if frame_ids is not None else list(range(rgbs.shape[0]))
        flows, past_flows, mask_bwds = self.load_flow(frame_ids=frame_ids, future_flow=True, past_flow=True, gts=rgbs)

        flow_ids = get_flowid(rgbs, flows, mask_bwds, rgb_threshold=rgb_threshold)

        self.n_frames = rgbs.shape[0]
        self.unq_inv = voxelization(flow_ids.reshape(-1), 
                                    rgbs.permute(0, 2, 3, 1).reshape(-1, 3), 
                                    None, None)

        return rgbs, None, None, flows, past_flows, mask_bwds
    
    @torch.no_grad()
    def load_flow(self, frame_ids=None, future_flow=False, past_flow=False, gts=None, save_flow=True, diff_threshold=0.1):
        flows, past_flows = [], []

        if self.flow_model.lower() == 'raft':
            model = eu.prepare_raft_model(self.device)
        elif self.flow_model.lower() == 'memflow':
            model = eu.prepare_memflow_model(self.device)
            gts = gts * 2.0 - 1.0
            future_flow_prev, past_flow_prev = None, None
        else:
            raise NotImplementedError(f"{self.flow_model} is not implemented yet.")

        if self.rgb_path.endswith(".mp4"):
            future_flow_path = self.rgb_path.replace(".mp4", f"_future_flow_{self.flow_model.lower()}")
            past_flow_path = self.rgb_path.replace(".mp4", f"_past_flow_{self.flow_model.lower()}")
        elif self.rgb_path.endswith(".gif"):
            future_flow_path = self.rgb_path.replace(".gif", f"_future_flow_{self.flow_model.lower()}")
            past_flow_path = self.rgb_path.replace(".gif", f"_past_flow_{self.flow_model.lower()}")
        else:
            future_flow_path = os.path.join(os.path.dirname(self.rgb_path), f"future_flow_{self.flow_model.lower()}")
            past_flow_path = os.path.join(os.path.dirname(self.rgb_path), f"past_flow_{self.flow_model.lower()}")
        
        if not os.path.exists(future_flow_path):
            os.makedirs(future_flow_path)
        if not os.path.exists(past_flow_path):
            os.makedirs(past_flow_path)
        
        if save_flow:
            print(f"[INFO] Saving future flows to {future_flow_path} as .pt files")
            print(f"[INFO] Saving past flows to {past_flow_path} as .pt files")
        
        for idx in tqdm(range(len(gts)), desc="Loading Flows"):
            if future_flow:
                if os.path.exists(os.path.join(future_flow_path, "{:04d}.pt".format(frame_ids[idx]))):
                    flow_fwd = torch.load(os.path.join(future_flow_path, "{:04d}.pt".format(frame_ids[idx])))
                else:
                    if idx == gts.shape[0] - 1:
                        flow_fwd = torch.zeros_like(gts[0:1, :2])
                    else:
                        padder = eu.InputPadder(gts[idx:idx+1].shape)
                        gt, gt_next = padder.pad(gts[idx:idx+1], gts[idx+1:idx+2])
                        if self.flow_model.lower() == 'raft':
                            _, flow_fwd = model(gt, gt_next, iters=20, test_mode=True)
                        else:
                            input = torch.cat([gt, gt_next], dim=0)
                            flow_low, flow_pre = model.step(input[None], flow_init=future_flow_prev)
                            flow_fwd = padder.unpad(flow_pre).cpu()
                            future_flow_prev = eu.forward_interpolate(flow_low[0])[None].cuda()
                    if save_flow:
                        torch.save(flow_fwd.cpu(), os.path.join(future_flow_path, "{:04d}.pt".format(frame_ids[idx])))

                flows.append(flow_fwd[0].cpu())

            if past_flow:
                if os.path.exists(os.path.join(past_flow_path, "{:04d}.pt".format(frame_ids[idx]))):
                    flow_bwd = torch.load(os.path.join(past_flow_path, "{:04d}.pt".format(frame_ids[idx])))
                else:
                    if idx == 0:
                        flow_bwd = torch.zeros_like(gts[0:1, :2])
                    else:
                        padder = eu.InputPadder(gts[idx:idx+1].shape)
                        gt, gt_prev = padder.pad(gts[idx:idx+1], gts[idx-1:idx])
                        if self.flow_model.lower() == 'raft':
                            _, flow_bwd = model(gt, gt_prev, iters=20, test_mode=True)
                        else:
                            input = torch.cat([gt, gt_prev], dim=0)
                            flow_low, flow_pre = model.step(input[None], flow_init=past_flow_prev)
                            flow_bwd = padder.unpad(flow_pre).cpu()
                            past_flow_prev = eu.forward_interpolate(flow_low[0])[None].cuda()
                    if save_flow:
                        torch.save(flow_bwd.cpu(), os.path.join(past_flow_path, "{:04d}.pt".format(frame_ids[idx])))

                past_flows.append(flow_bwd[0].cpu())

        del model  # Free up memory

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
            mask_bwds = get_mask_bwds(gts, flows, past_flows, alpha=self.alpha, diff_threshold=diff_threshold)
        else:
            mask_bwds = None

        return flows, past_flows, mask_bwds