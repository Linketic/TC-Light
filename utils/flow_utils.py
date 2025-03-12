import torch
import torch.nn.functional as F
from evaluation import eval_utils as eu

def warp_flow(frames, past_flows):
    
    N, _, H, W = frames.shape
    flow_new = past_flows[:, :2].clone()
    flow_new[:, 0, :, :] += torch.arange(W, device=flow_new.device)
    flow_new[:, 1, :, :] += torch.arange(H, device=flow_new.device)[:, None]
    # resides flow_new to [-1, 1]
    flow_new[:, 0] = (flow_new[:, 0] / (W - 1) - 0.5) * 2
    flow_new[:, 1] = (flow_new[:, 1] / (H - 1) - 0.5) * 2
    frame_warp = F.grid_sample(frames, flow_new.permute(0, 2, 3, 1), mode='bicubic', padding_mode='zeros', align_corners=True)

    return frame_warp

def compute_fwdbwd_mask(fwd_flow, bwd_flow, alpha=0.1):

    bwd2fwd_flow = warp_flow(bwd_flow, fwd_flow)
    fwd_lr_error = torch.linalg.norm(fwd_flow + bwd2fwd_flow, dim=1)
    fwd_mask = fwd_lr_error < alpha * (torch.linalg.norm(fwd_flow, dim=1) + torch.linalg.norm(bwd2fwd_flow, dim=1)) + alpha

    fwd2bwd_flow = warp_flow(fwd_flow, bwd_flow)
    bwd_lr_error = torch.linalg.norm(bwd_flow + fwd2bwd_flow, dim=1)
    bwd_mask = bwd_lr_error < alpha * (torch.linalg.norm(bwd_flow, dim=1) + torch.linalg.norm(fwd2bwd_flow, dim=1)) + alpha

    return fwd_mask, bwd_mask

def get_mask_bwds(org_images, flows, past_flows, alpha=0.1, diff_threshold=0.1):

    mask_bwds = torch.ones_like(org_images[:, 0], dtype=torch.bool)
    _, mask_bwds[1:] = compute_fwdbwd_mask(flows[:-1], past_flows[1:], alpha=alpha)
    org_images_warp = warp_flow(org_images[:-1], past_flows[1:])
    mask_bwds[1:] &= (org_images_warp - org_images[1:]).abs().max(dim=1).values < org_images.max().item() * diff_threshold
    mask_bwds = mask_bwds[:, None, ...].repeat(1, 3, 1, 1)

    return mask_bwds