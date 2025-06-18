import os
import cv2
import torch
from tqdm import tqdm

from evaluation import eval_utils as eu
from utils.VidToMe import load_video as _load_video
from utils.general_utils import voxelization, process_frames
from utils.flow_utils import get_soft_mask_bwds, get_flowid


class VideoDataParser:
    def __init__(self, data_config, device="cuda", dtype=torch.float32):
        self.rgb_path = data_config.rgb_path
        self.fps = getattr(data_config, "fps", 30)
        self.alpha = getattr(data_config, "alpha", 0.5)
        self.flow_model = getattr(data_config, "flow_model", "memflow")
        self.h, self.w = data_config.height, data_config.width
        self.voxel_size = None
        self.device = device
        self.dtype = dtype
        self.unq_inv = None

        if self.rgb_path.endswith(('.mp4', '.gif', '.avi')):
            cap = cv2.VideoCapture(self.rgb_path)
            self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            self.n_frames = len([
                name for name in os.listdir(self.rgb_path)
                if os.path.isfile(os.path.join(self.rgb_path, name))
            ])

    @torch.no_grad()
    def load_video(self, frame_ids=None, path=None):
        path = path or self.rgb_path
        rgbs = _load_video(path, self.h, self.w, frame_ids=frame_ids, device=self.device, base=8)
        if rgbs.min() < 0:
            rgbs = (rgbs + 1.0) * 127.0 / 255.0

        self.n_frames = rgbs.shape[0]
        return rgbs

    @torch.no_grad()
    def load_data(self, frame_ids=None, rgb_threshold=0.01):
        rgbs = _load_video(self.rgb_path, self.h, self.w, frame_ids=frame_ids, device='cpu', base=8)
        if rgbs.min() < 0:
            rgbs = (rgbs + 1.0) * 127.0 / 255.0

        self.n_frames = rgbs.shape[0]
        frame_ids = frame_ids or list(range(rgbs.shape[0]))

        future_flows, past_flows, mask_bwds = self.load_flow(frame_ids, True, True, rgbs)
        flow_ids = get_flowid(rgbs, future_flows, mask_bwds, rgb_threshold=rgb_threshold).view(-1, 1)

        rgbs = rgbs.permute(0, 2, 3, 1).reshape(-1, 3).to(self.device)
        torch.cuda.empty_cache()

        self.unq_inv = voxelization(flow_ids, rgbs, None, None)
        torch.cuda.empty_cache()

        return rgbs, None, None, future_flows, past_flows, mask_bwds

    @torch.no_grad()
    def load_flow(self, frame_ids=None, future_flow=False, past_flow=False, gts=None,
                  target_ids=None, save_flow=True, diff_threshold=0.1):
        flows, past_flows = [], []
        future_flow_prev, past_flow_prev = None, None

        if target_ids is not None:
            assert len(target_ids) == len(frame_ids), "target_ids and frame_ids must match"

        model = {
            'raft': eu.prepare_raft_model,
            'memflow': eu.prepare_memflow_model
        }.get(self.flow_model.lower())(self.device)

        if self.flow_model.lower() == 'memflow':
            gts = gts * 2.0 - 1.0

        gts = gts.to(self.device, dtype=self.dtype)
        future_flow_path = self.create_folder(f"future_flow_{self.flow_model.lower()}")
        past_flow_path = self.create_folder(f"past_flow_{self.flow_model.lower()}")

        if save_flow:
            print(f"[INFO] Saving future flows to {future_flow_path} as .pt files")
            print(f"[INFO] Saving past flows to {past_flow_path} as .pt files")

        for idx in tqdm(range(len(gts)), desc="Loading Flows"):
            if future_flow:
                flow_fwd = self.load_or_calc_flow(
                    idx, gts, frame_ids, model, future_flow_prev, True,
                    future_flow_path, save_flow
                )
                flows.append(flow_fwd[0].cpu())

            if past_flow:
                flow_bwd = self.load_or_calc_flow(
                    idx, gts, frame_ids, model, past_flow_prev, False,
                    past_flow_path, save_flow
                )
                past_flows.append(flow_bwd[0].cpu())

        del model

        flows = self.process_flow(flows).to(self.device, dtype=self.dtype) if future_flow else None
        past_flows = self.process_flow(past_flows).to(self.device, dtype=self.dtype) if past_flow else None
        mask_bwds = get_soft_mask_bwds(gts, flows, past_flows, alpha=self.alpha,
                                       diff_threshold=diff_threshold) if future_flow and past_flow else None

        return flows, past_flows, mask_bwds

    def load_or_calc_flow(self, idx, gts, frame_ids, model, flow_prev, is_future, path, save):
        fname = os.path.join(path, f"{frame_ids[idx]:04d}.pt")
        if os.path.exists(fname):
            return torch.load(fname)

        zero_idx = gts.shape[0] - 1 if is_future else 0
        src = gts[idx:idx + 1]
        tgt = gts[idx + 1:idx + 2] if is_future else gts[idx - 1:idx]

        flow, flow_prev = self.calc_flow(idx, zero_idx, src, tgt, model, flow_prev)
        if save:
            torch.save(flow.cpu(), fname)

        return flow

    def create_folder(self, name):
        ext = os.path.splitext(self.rgb_path)[-1]
        base_path = self.rgb_path.replace(ext, f"_{name}") if not os.path.isdir(self.rgb_path) else os.path.join(self.rgb_path, name)
        os.makedirs(base_path, exist_ok=True)
        return base_path

    def process_flow(self, flow_list):
        flow = torch.stack(flow_list)
        N, _, H, W = flow.shape
        flow = process_frames(flow, self.h, self.w)
        scale = max(self.w / W, self.h / H)
        return flow * scale

    def calc_flow(self, idx, zero_idx, src, tgt, model, flow_prev=None):
        if idx == zero_idx:
            return torch.zeros_like(src[:, :2]), flow_prev

        padder = eu.InputPadder(src.shape)
        src, tgt = padder.pad(src, tgt)

        if self.flow_model.lower() == 'raft':
            _, flow = model(src, tgt, iters=20, test_mode=True)
        else:
            input_pair = torch.cat([src, tgt], dim=0)
            flow_low, flow_pre = model.step(input_pair[None], flow_init=flow_prev)
            flow = padder.unpad(flow_pre).cpu()
            flow_prev = eu.forward_interpolate(flow_low[0])[None].cuda()

        return flow, flow_prev
