from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import torch.nn.functional as F
import cv2
import imageio
import argparse
import torch
import clip
import lpips
import numpy as np
import os
from glob import glob
from math import exp
from skimage.metrics import structural_similarity

from torch.autograd import Variable
from evaluation.core.raft import RAFT
from evaluation.core.utils.utils import InputPadder, forward_interpolate

def get_frame_ids(frame_range, frame_ids=None):
    if frame_ids is None:
        frame_ids = list(range(*frame_range))
    frame_ids = sorted(frame_ids)

    if len(frame_ids) > 4:
        frame_ids_str = "{} {} ... {} {}".format(
            *frame_ids[:2], *frame_ids[-2:])
    else:
        frame_ids_str = " ".join(["{}"] * len(frame_ids)).format(*frame_ids)
    print("[INFO] frame indexes: ", frame_ids_str)
    return frame_ids

def video_to_pil_list(video_path):
    pil_list = []
    if video_path.endswith('.mp4'):
        vidcap = cv2.VideoCapture(video_path)
        while True:
            success, image = vidcap.read()
            if success:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_list.append(Image.fromarray(image))
            else:
                break
    elif video_path.endswith('.gif'):
        gif = imageio.get_reader(video_path)

        for frame in gif:
            pil_list.append(Image.fromarray(frame))
    else:
        frame_paths = []
        for ext in [".jpg", ".png"]:
            frame_paths += glob(os.path.join(video_path, f"*{ext}"))
        frame_paths = sorted(frame_paths)
        for frame_path in frame_paths:
            frame = Image.open(frame_path).convert('RGB')
            pil_list.append(frame)
    
    return pil_list

def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid


def bilinear_sample(img,
                    sample_coords,
                    mode='bilinear',
                    padding_mode='zeros',
                    return_mask=False):
    # img: [B, C, H, W]
    # sample_coords: [B, 2, H, W] in image scale
    if sample_coords.size(1) != 2:  # [B, H, W, 2]
        sample_coords = sample_coords.permute(0, 3, 1, 2)

    b, _, h, w = sample_coords.shape

    # Normalize to [-1, 1]
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1

    grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, 2]

    img = F.grid_sample(img,
                        grid,
                        mode=mode,
                        padding_mode=padding_mode,
                        align_corners=True)

    if return_mask:
        mask = (x_grid >= -1) & (y_grid >= -1) & (x_grid <= 1) & (
            y_grid <= 1)  # [B, H, W]

        return img, mask

    return img


def flow_warp_rerender(feature,
              flow,
              mask=False,
              mode='bilinear',
              padding_mode='zeros'):
    b, c, h, w = feature.size()
    assert flow.size(1) == 2

    grid = coords_grid(b, h, w).to(flow.device) + flow  # [B, 2, H, W]

    return bilinear_sample(feature,
                           grid,
                           mode=mode,
                           padding_mode=padding_mode,
                           return_mask=mask)


def clip_text(pil_list, text_prompt, preprocess, device, model):
    text = clip.tokenize([text_prompt]).to(device)

    scores = []
    images = []
    with torch.no_grad():
        text_features = model.encode_text(text)
        for pil in pil_list:
            image = preprocess(pil).unsqueeze(0).to(device)
            images.append(image)
        image_features = model.encode_image(torch.cat(images))
        scores = [torch.cosine_similarity(text_features, image_feature).item() for image_feature in image_features]

    score = sum(scores) / len(scores)
    
    return score

def clip_frame(pil_list, preprocess, device, model):
    image_features = []
    images = []
    with torch.no_grad():
        for pil in pil_list:
            image = preprocess(pil).unsqueeze(0).to(device)
            images.append(image)
        
        image_features = model.encode_image(torch.cat(images))
        
    image_features = image_features.cpu().numpy()
    cosine_sim_matrix = cosine_similarity(image_features)
    np.fill_diagonal(cosine_sim_matrix, 0)  # set diagonal elements to 0
    score = cosine_sim_matrix.sum() / (len(pil_list) * (len(pil_list)-1))

    return score

def pick_score_func(frames, prompt, model, processor, device):
    image_inputs = processor(images=frames, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
    text_inputs = processor(text=prompt, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)

    with torch.no_grad():
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        score_per_image = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        score_per_image = score_per_image.detach().cpu().numpy()
        score = score_per_image.mean()

    return score

def prepare_raft_model(device):
    raft_dict = {
        'model': '/data1/yang_liu/python_workspace/RAVE/pretrained_models/raft/raft-things.pth',
        'small': False,
        'mixed_precision': False,
        'alternate_corr': False
    }

    args = argparse.Namespace(**raft_dict)

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, weights_only=False))

    model = model.module
    model.to(device)
    model.eval()

    return model

def prepare_memflow_model(device):
    memflow_dict = {
        'name': 'MemFlowNet',
        'stage': 'things',
        'restore_ckpt': "/data1/yang_liu/python_workspace/IC-Light/models/memflow/MemFlowNet_things.pth",
    }

    args = argparse.Namespace(**memflow_dict)

    if args.name == "MemFlowNet":
        if args.stage == 'things':
            from evaluation.memflow.configs.things_memflownet import get_cfg
        elif args.stage == 'sintel':
            from evaluation.memflow.configs.sintel_memflownet import get_cfg
        elif args.stage == 'spring_only':
            from evaluation.memflow.configs.spring_memflownet import get_cfg
        elif args.stage == 'kitti':
            from evaluation.memflow.configs.kitti_memflownet import get_cfg
        else:
            raise NotImplementedError
    elif args.name == "MemFlowNet_T":
        if args.stage == 'things':
            from evaluation.memflow.configs.things_memflownet_t import get_cfg
        elif args.stage == 'things_kitti':
            from evaluation.memflow.configs.things_memflownet_t_kitti import get_cfg
        elif args.stage == 'sintel':
            from evaluation.memflow.configs.sintel_memflownet_t import get_cfg
        elif args.stage == 'kitti':
            from evaluation.memflow.configs.kitti_memflownet_t import get_cfg
        else:
            raise NotImplementedError

    cfg = get_cfg()
    cfg.update(vars(args))

    from evaluation.memflow.core.Networks import build_network
    from evaluation.memflow.inference import inference_core_skflow as inference_core
    model = build_network(cfg).to(device)

    if cfg.restore_ckpt is not None:
        print("[Loading ckpt from {}]".format(cfg.restore_ckpt))
        ckpt = torch.load(cfg.restore_ckpt, map_location='cpu', weights_only=True)
        ckpt_model = ckpt['model'] if 'model' in ckpt else ckpt
        if 'module' in list(ckpt_model.keys())[0]:
            for key in ckpt_model.keys():
                ckpt_model[key.replace('module.', '', 1)] = ckpt_model.pop(key)
            model.load_state_dict(ckpt_model, strict=True)
        else:
            model.load_state_dict(ckpt_model, strict=True)

    model.eval()
    processor = inference_core.InferenceCore(model, config=cfg)

    return processor

def load_image(imfile, DEVICE, dtype=torch.float):
    img = np.array(imfile).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).to(dtype)
    return img[None].to(DEVICE)

def resize_flow(flow, img_h, img_w):
    flow_h, flow_w = flow.shape[0], flow.shape[1]
    flow[:, :, 0] *= float(img_w) / float(flow_w)
    flow[:, :, 1] *= float(img_h) / float(flow_h)
    flow = cv2.resize(flow, (img_w, img_h), cv2.INTER_LINEAR)

    return flow

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow_new = flow.copy()
    flow_new[:, :, 0] += np.arange(w)
    flow_new[:, :, 1] += np.arange(h)[:, np.newaxis]

    res = cv2.remap(
        img, flow_new, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT
    )  # because of interpolation, the warpped result woule never be perfectly accurate
    return res

def compute_fwdbwd_mask(fwd_flow, bwd_flow):
    alpha_1 = 0.5
    alpha_2 = 0.5

    bwd2fwd_flow = warp_flow(bwd_flow, fwd_flow)
    fwd_lr_error = np.linalg.norm(fwd_flow + bwd2fwd_flow, axis=-1)
    fwd_mask = (
        fwd_lr_error
        < alpha_1
        * (np.linalg.norm(fwd_flow, axis=-1) + np.linalg.norm(bwd2fwd_flow, axis=-1))
        + alpha_2
    )

    fwd2bwd_flow = warp_flow(fwd_flow, bwd_flow)
    bwd_lr_error = np.linalg.norm(bwd_flow + fwd2bwd_flow, axis=-1)

    bwd_mask = (
        bwd_lr_error
        < alpha_1
        * (np.linalg.norm(bwd_flow, axis=-1) + np.linalg.norm(fwd2bwd_flow, axis=-1))
        + alpha_2
    )
    return fwd_mask, bwd_mask

def SaveWarpingImage(edit_pil_list, source_pil_list, raft_model, device, distance_func, flow_fwd_list=None, flow_bwd_list=None):

    img_w, img_h = edit_pil_list[0].size

    ssim_list = []
    for idx, pil_img in enumerate(edit_pil_list[:-1]):

        pil_img = load_image(pil_img, device)
        gt = load_image(source_pil_list[idx], device)

        next_idx = idx+1
        pil_next_img = load_image(edit_pil_list[next_idx], device)
        gt_next = load_image(source_pil_list[next_idx], device)
        
        padder = InputPadder(gt.shape)
        gt, gt_next = padder.pad(gt, gt_next)
        pil_img, pil_next_img = padder.pad(pil_img, pil_next_img)

        if flow_fwd_list is None or flow_bwd_list is None:
            _, flow_fwd = raft_model(gt, gt_next, iters=20, test_mode=True)
            _, flow_bwd = raft_model(gt_next, gt, iters=20, test_mode=True)
        else:
            flow_fwd = flow_fwd_list[idx:idx+1, :2]
            flow_bwd = flow_bwd_list[idx+1:idx+2, :2]
        flow_fwd = padder.unpad(flow_fwd[0]).detach().cpu().numpy().transpose(1, 2, 0)
        flow_bwd = padder.unpad(flow_bwd[0]).detach().cpu().numpy().transpose(1, 2, 0)

        pil_img = padder.unpad(pil_img[0]).detach().cpu().numpy().transpose(1, 2, 0)
        pil_next_img = padder.unpad(pil_next_img[0]).detach().cpu().numpy().transpose(1, 2, 0)

        flow_fwd = resize_flow(flow_fwd, img_h, img_w)
        flow_bwd = resize_flow(flow_bwd, img_h, img_w)

        pil_img = resize_flow(pil_img, img_h, img_w)
        pil_next_img = resize_flow(pil_next_img, img_h, img_w)

        _, mask_bwd = compute_fwdbwd_mask(flow_fwd, flow_bwd)

        warped_frame = warp_flow(pil_img, flow_bwd)
        warped_zero = np.zeros_like(warped_frame)

        # warping the mask
        mask_bwd = np.expand_dims(mask_bwd, axis=-1)
        warped_frame = np.where(mask_bwd, warped_frame, warped_zero)
        pil_next_img = np.where(mask_bwd, pil_next_img, warped_zero)

        if distance_func == structural_similarity:
            ssim = distance_func(np.uint8(warped_frame), np.uint8(pil_next_img), channel_axis=2)
        else:
            ssim = distance_func(np.uint8(warped_frame), np.uint8(pil_next_img))
        
        # print(f'Idx {idx}: {ssim}')
        ssim_list.append(ssim)

        # warped_frame = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2RGB)
        # pil_next_img = cv2.cvtColor(pil_next_img, cv2.COLOR_BGR2RGB)

        # import os
        # if not os.path.exists('warped'):
        #     os.makedirs('warped')
        # if not os.path.exists('warped_gt'):
        #     os.makedirs('warped_gt')
        
        # cv2.imwrite(os.path.join('warped', f'{idx}.png'), warped_frame)
        # cv2.imwrite(os.path.join('warped_gt', f'{idx}.png'), pil_next_img)

    return np.mean(ssim_list)

@torch.no_grad()
def FrameLPIPS(edit_pil_list, source_pil_list, device):

    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()

    lpips_list = []
    for idx, pil_img in enumerate(edit_pil_list[:-1]):

        pil_img = load_image(pil_img, device)
        gt = load_image(source_pil_list[idx], device)
        
        padder = InputPadder(gt.shape)
        gt, pil_img = padder.pad(gt, pil_img)

        lpips_metric = loss_fn_vgg(gt, pil_img).squeeze()
        
        lpips_list.append(lpips_metric)

    return torch.mean(torch.stack(lpips_list)).item()

loss_fn_vgg = None

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

@torch.no_grad()
def lpips_func(img1, img2, net_type='vgg'):

    global loss_fn_vgg
    if loss_fn_vgg is None:
        loss_fn_vgg = lpips.LPIPS(net=net_type).cuda(device=img1.device)

    return loss_fn_vgg(img1, img2).squeeze()