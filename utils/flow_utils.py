import torch
import torch.nn.functional as F
from tqdm import tqdm

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
    mask_bwds = (-torch.nn.MaxPool2d(kernel_size=5, stride=1, padding=2)(-mask_bwds.float())).bool()  # dilate error area to enhance robustness

    return mask_bwds

def get_soft_mask_bwds(org_images, flows, past_flows, alpha=0.1, beta=1e2, diff_threshold=0.1, batch_size=64):

    mask_bwds = torch.ones_like(org_images[:, 0])

    for i in tqdm(range(0, len(flows) - 1, batch_size), desc="Computing soft masks"):
        fwd2bwd_flow = warp_flow(flows[:-1][i:i+batch_size], past_flows[1:][i:i+batch_size])
        mask_bwds[i+1:i+1+batch_size] *= torch.sigmoid(-beta * (torch.linalg.norm(past_flows[1:][i:i+batch_size] + fwd2bwd_flow, dim=1) - 
                     ((torch.linalg.norm(past_flows[1:][i:i+batch_size], dim=1) + torch.linalg.norm(fwd2bwd_flow, dim=1)) + 1) * alpha))

        diff_images_warp = warp_flow(org_images[:-1][i:i+batch_size], past_flows[1:][i:i+batch_size])
        diff_images_warp -= org_images[1:][i:i+batch_size]
        diff_images_warp = diff_images_warp.abs_().max(dim=1).values
        mask_bwds[i+1:i+1+batch_size] *= torch.sigmoid(-beta * (diff_images_warp - org_images.max().item() * diff_threshold))
    
    return mask_bwds[:, None]

def get_flowid(frames, flows, mask_bwds, rgb_threshold=0.01):
    N, _, H, W = frames.shape
    frames = frames.to(device=flows.device, dtype=flows.dtype)
    # automatically choose dtype according to N*H*W
    if N * H * W < 2**31:
        int_dtype = torch.int32
    else:
        int_dtype = torch.int64
    flow_ids = torch.ones_like(frames[:, 0], dtype=int_dtype) * -1
    flow_ids[0] = torch.arange(H * W).view(H, W)
    last_id = H * W

    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W))
    grid_y = grid_y.to(device=flows.device)
    grid_x = grid_x.to(device=flows.device)
    diff_threshold = frames.max().item() * rgb_threshold
    for i in tqdm(range(1, N), desc="Assigning flow ids"):
        x = (grid_x + flows[i-1, 0]).round().to(int_dtype)
        y = (grid_y + flows[i-1, 1]).round().to(int_dtype)
        mask = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        
        # validate correctness
        # proj_frame = torch.zeros_like(frames[i])
        # proj_frame[:, y[mask], x[mask]] = frames[i-1, :, grid_y[mask], grid_x[mask]]
        # torchvision.utils.save_image(proj_frame, "projeted2cur.png")
        # torchvision.utils.save_image(frames[i], "cur.png")

        mask &= (mask_bwds[i, 0] > 0.5)
        
        # cut off flow when error is significant
        diff_mask = (frames[i, :, y[mask], x[mask]] - frames[i-1, :, grid_y[mask], grid_x[mask]]).abs().max(dim=0).values < diff_threshold
        flow_ids[i, y[mask][diff_mask], x[mask][diff_mask]] = flow_ids[i-1, grid_y[mask][diff_mask], grid_x[mask][diff_mask]]

        unassigned = (flow_ids[i] == -1)
        flow_ids[i, unassigned] = last_id + torch.arange(unassigned.sum(), device=frames.device, dtype=int_dtype)
        last_id += unassigned.sum()
    
    return flow_ids