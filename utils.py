#!/usr/bin/env python
import argparse

from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import torch.nn as nn
import torch.nn.functional as F

class MRI_Dataset(Dataset):
    def __init__(self, t1_path, t2_path, flair_path, mode='test'):
        self.mode = mode
        self.t1_path = Path(t1_path)
        self.t2_path = Path(t2_path)
        self.flair_path = Path(flair_path)

        self.mprage_paths = [self.t1_path]
        self.t2_paths = [self.t2_path]
        self.flair_paths = [self.flair_path]
        
        if len(self.mprage_paths) == 0:
            raise RuntimeError("No T1w NIfTI files found.")
        if len(self.t2_paths) == 0:
            raise RuntimeError("No T2w NIfTI files found.")
        if len(self.flair_paths) == 0:
            raise RuntimeError("No FLAIR NIfTI files found.")
    
    def __len__(self):
        return len(self.mprage_paths)

    def __getitem__(self, idx):
        print(f"Loading image: {str(self.mprage_paths[idx])}")
        mprage_img = nib.load(str(self.mprage_paths[idx])).get_fdata().astype(np.float32)
        print(f"Loading image: {str(self.t2_paths[idx])}")
        t2_img = nib.load(str(self.t2_paths[idx])).get_fdata().astype(np.float32)
        print(f"Loading image: {str(self.flair_paths[idx])}")
        flair_img = nib.load(str(self.flair_paths[idx])).get_fdata().astype(np.float32)

        # Axial: (D, H, W)
        axial_slices = np.stack([mprage_img, t2_img, flair_img], axis=0)  # shape: (3, D, H, W)
        axial_tensors = torch.tensor(axial_slices, dtype=torch.float32)

        # Sagittal: (2, 0, 1) → (W, D, H)
        t1_sagittal  = np.transpose(mprage_img, (2, 0, 1))
        t2_sagittal  = np.transpose(t2_img,    (2, 0, 1))
        flair_sagittal = np.transpose(flair_img, (2, 0, 1))
        sagittal_slices = np.stack([t1_sagittal, t2_sagittal, flair_sagittal], axis=0)  # shape: (3, W, D, H)
        sagittal_tensors = torch.tensor(sagittal_slices, dtype=torch.float32)

        # Coronal: (1, 2, 0) → (H, W, D)
        t1_coronal  = np.transpose(mprage_img, (1, 2, 0))
        t2_coronal  = np.transpose(t2_img,    (1, 2, 0))
        flair_coronal = np.transpose(flair_img, (1, 2, 0))
        coronal_slices = np.stack([t1_coronal, t2_coronal, flair_coronal], axis=0)  # shape: (3, H, W, D)
        coronal_tensors = torch.tensor(coronal_slices, dtype=torch.float32)

        return {
            'axial': axial_tensors,       # (3, D, H, W)
            'sagittal': sagittal_tensors, # (3, W, D, H)
            'coronal': coronal_tensors    # (3, H, W, D)
        }

class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, conditional_ch=0, num_lvs=4, base_ch=16, final_act='sigmoid'):
        super().__init__()
        self.final_act = final_act
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, 1, 1)

        self.down_convs = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for lv in range(num_lvs):
            ch = base_ch * (2 ** lv)
            self.down_convs.append(ConvBlock2d(ch + conditional_ch, ch * 2, ch * 2))
            self.down_samples.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.up_samples.append(Upsample(ch * 4))
            self.up_convs.append(ConvBlock2d(ch * 4, ch * 2, ch * 2))
        bottleneck_ch = base_ch * (2 ** num_lvs)
        self.bottleneck_conv = ConvBlock2d(bottleneck_ch, bottleneck_ch * 2, bottleneck_ch * 2)
        self.out_conv = nn.Sequential(nn.Conv2d(base_ch * 2, base_ch, 3, 1, 1),
                                      nn.LeakyReLU(0.1),
                                      nn.Conv2d(base_ch, out_ch, 3, 1, 1))
        
    def forward(self, in_tensor, condition=None):
        encoded_features = []
        x = self.in_conv(in_tensor)
        for down_conv, down_sample in zip(self.down_convs, self.down_samples):
            if condition is not None:
                feature_dim = x.shape[-1]
                down_conv_out = down_conv(torch.cat([x, condition.repeat(1, 1, feature_dim, feature_dim)], dim=1))
            else:
                down_conv_out = down_conv(x)
            x = down_sample(down_conv_out)
            encoded_features.append(down_conv_out)
        x = self.bottleneck_conv(x)
        for encoded_feature, up_conv, up_sample in zip(reversed(encoded_features),
                                                       reversed(self.up_convs),
                                                       reversed(self.up_samples)):
            x = up_sample(x, encoded_feature)
            x = up_conv(x)
        x = self.out_conv(x)
        if self.final_act == 'sigmoid':
            x = torch.sigmoid(x)
            x = x*5000
        elif self.final_act == "relu":
            x = torch.relu(x)
        elif self.final_act == 'tanh':
            x = torch.tanh(x)
        else:
            x = x
        return x

class ConvBlock2d(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, 1, 1),
            nn.InstanceNorm2d(mid_ch),
            nn.LeakyReLU(0.1),
            nn.Conv2d(mid_ch, out_ch, 3, 1, 1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.1)
        )

    def forward(self, in_tensor):
        return self.conv(in_tensor)


class Upsample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        out_ch = in_ch // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.1)
        )

    def forward(self, in_tensor, encoded_feature):
        up_sampled_tensor = F.interpolate(in_tensor, size=None, scale_factor=2, mode='bilinear', align_corners=False)
        up_sampled_tensor = self.conv(up_sampled_tensor)
        return torch.cat([encoded_feature, up_sampled_tensor], dim=1)

class MultiTICalc(nn.Module):
    def __init__(self):
        super(MultiTICalc, self).__init__()

    @staticmethod
    def zinf(vals):
        """Replace non-finite values with 0."""
        vals = vals.clone()
        vals[~torch.isfinite(vals)] = 0
        return vals

    def forward(self, t1map, pdmap, ti, tr):
        """Calculate synthetic T1w image from pdmap and t1map."""
        eps = 1e-8
        t1map_safe = t1map + eps
        ti_img = pdmap * (
            1 - 2 * torch.exp(-ti / t1map_safe) + torch.exp(-tr / t1map_safe)
        )
        ti_img = self.zinf(torch.abs(ti_img))
        # Keep dimensions like (batch, channel, H, W); adjust unsqueeze as needed
        return ti_img
