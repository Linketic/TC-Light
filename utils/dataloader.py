import os
import math
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
import torch.nn.functional as F

class OptDataset(Dataset):
    def __init__(self, edited_images, past_flows, mask_bwd, device):

        super(OptDataset, self).__init__()

        self.edited_images = edited_images
        self.past_flows = past_flows
        self.mask_bwd = mask_bwd
        self.device = device

        if self.edited_images.max() > 1:
            self.edited_images = self.edited_images / 255.0  # normalise to [0, 1] for further optimization

        # initialise the linalg module for lazy loading
        torch.inverse(torch.ones((1, 1), device=self.device))

    def __len__(self):
        return len(self.edited_images)

    def __getitem__(self, idx):

        edited_image = self.edited_images[idx]
        past_flow = self.past_flows[idx]
        mask_bwd = self.mask_bwd[idx]
        pre_edited_image = self.edited_images[idx - 1] if idx > 0 else edited_image

        return idx, edited_image, pre_edited_image, past_flow, mask_bwd
    
    @torch.no_grad()
    def exposure_align(self, exposure):
        N, _, H, W = self.edited_images.shape
        self.edited_images = torch.bmm(self.edited_images.permute(0, 2, 3, 1).reshape(N, H*W, -1), exposure[:, :3, :3]) + exposure[:, None, :3, 3]
        self.edited_images = torch.clamp(self.edited_images, 0, 1).reshape(N, H, W, 3).permute(0, 3, 1, 2)  # N x 3 x H x W
