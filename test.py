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

from utils import UNet, MRI_Dataset, MultiTICalc

# Function to load the dataset and perform the test operation
def run_testing(t1: str, t2: str, flair: str, ti: int, o: str, weights: str, gpu: int = 0):
    # Setup the model
    model = UNet(in_ch=3, out_ch=2).to(f'cuda:{gpu}')
    # checkpoint = torch.load('/opt/run/models/best_unet_epoch426.pth')  # Load your trained model
    checkpoint = torch.load(Path(weights))
    model.load_state_dict(checkpoint['unet'])

    # Your input and output paths
    t1_DataPath = Path(t1)
    t2_DataPath = Path(t2)
    flair_DataPath = Path(flair)
    output_path = Path(o)

    # Function to load dataset
    dataset = MRI_Dataset(t1_DataPath, t2_DataPath, flair_DataPath)  # load your dataset
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # Load header and affine from the first image in the dataset
    first_img_path = dataset.mprage_paths[0]
    header = nib.load(str(first_img_path)).header
    affine = nib.load(str(first_img_path)).affine
    filename = str(first_img_path.name).replace('.nii.gz','_SYN')
    # print(filename)

    model.eval()
    axial_volumes = []
    sagittal_volumes = []
    coronal_volumes = []

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            axial_slices = batch['axial'].to(f'cuda:{gpu}')
            sagittal_slices = batch['sagittal'].to(f'cuda:{gpu}')
            coronal_slices = batch['coronal'].to(f'cuda:{gpu}')

            # Process axial slices
            axial_results = []
            for i in range(axial_slices.shape[2]):
                axial_slice = axial_slices[:, :, i, :]  # (1, 3, H, W)
                axial_result = model(axial_slice)  # (1, 2, H, W)
                axial_result = axial_result.cpu().numpy().squeeze(0)  # (2, H, W)
                axial_results.append(axial_result)

            axial_volume = np.stack(axial_results, axis=0)  # (num_slices, 2, H, W)

            # Save channel 0: T1 map
            axial_t1map_volume = axial_volume[:, 0, :, :]  # (num_slices, H, W)
            axial_nifti_img_t1 = nib.Nifti1Image(axial_t1map_volume, affine, header)
            nib.save(axial_nifti_img_t1, Path(output_path) / f'{filename}_axial_t1map.nii.gz')

            # Save channel 1: Proton Density map
            axial_pdmap_volume = axial_volume[:, 1, :, :]  # (num_slices, H, W)
            axial_nifti_img_pd = nib.Nifti1Image(axial_pdmap_volume, affine, header)
            nib.save(axial_nifti_img_pd, Path(output_path) / f'{filename}_axial_pdmap.nii.gz')

            # Same for sagittal results
            sagittal_results = []
            for i in range(sagittal_slices.shape[2]):
                sagittal_slice = sagittal_slices[:, :, i, :]  # (1, 3, H, W)
                sagittal_result = model(sagittal_slice)  # (1, 2, H, W)
                sagittal_result = sagittal_result.cpu().numpy().squeeze(0)  # (2, H, W)
                sagittal_results.append(sagittal_result)

            sagittal_volume = np.stack(sagittal_results, axis=0)  # (num_slices, 2, H, W)

            # Save channel 0: T1 map
            sagittal_t1map_volume = sagittal_volume[:, 0, :, :]  # (num_slices, H, W)
            sagittal_t1map_volume = sagittal_t1map_volume.transpose(1,2,0)
            sagittal_nifti_img_t1 = nib.Nifti1Image(sagittal_t1map_volume, affine, header)
            nib.save(sagittal_nifti_img_t1, Path(output_path) / f'{filename}_sagittal_t1map.nii.gz')

            # Save channel 1: Proton Density map
            sagittal_pdmap_volume = sagittal_volume[:, 1, :, :]  # (num_slices, H, W)
            sagittal_pdmap_volume = sagittal_pdmap_volume.transpose(1,2,0)
            sagittal_nifti_img_pd = nib.Nifti1Image(sagittal_pdmap_volume, affine, header)
            nib.save(sagittal_nifti_img_pd, Path(output_path) / f'{filename}_sagittal_pdmap.nii.gz')

            # Process coronal results
            coronal_results = []
            for i in range(coronal_slices.shape[2]):
                coronal_slice = coronal_slices[:, :, i, :]  # (1, 3, H, W)
                coronal_result = model(coronal_slice)  # (1, 2, H, W)
                coronal_result = coronal_result.cpu().numpy().squeeze(0)  # (2, H, W)
                coronal_results.append(coronal_result)

            coronal_volume = np.stack(coronal_results, axis=0)  # (num_slices, 2, H, W)

            # Save channel 0: T1 map
            coronal_t1map_volume = coronal_volume[:, 0, :, :]  # (num_slices, H, W)
            coronal_t1map_volume = coronal_t1map_volume.transpose(2,0,1)

            coronal_nifti_img_t1 = nib.Nifti1Image(coronal_t1map_volume, affine, header)
            nib.save(coronal_nifti_img_t1, Path(output_path) / f'{filename}_coronal_t1map.nii.gz')

            # Save channel 1: Proton Density map
            coronal_pdmap_volume = coronal_volume[:, 1, :, :]  # (num_slices, H, W)
            coronal_pdmap_volume = coronal_pdmap_volume.transpose(2,0,1)
            coronal_nifti_img_pd = nib.Nifti1Image(coronal_pdmap_volume, affine, header)
            nib.save(sagittal_nifti_img_pd, Path(output_path) / f'{filename}_coronal_pdmap.nii.gz')

            print(f"Computing median volume...")
            
            # Median Volume
            # List of volumes, shape for each: (H, W, D)
            t1maps = [
                axial_t1map_volume,        # from axial
                sagittal_t1map_volume, # from sagittal
                coronal_t1map_volume   # from coronal
            ]
            pdmaps = [
                axial_pdmap_volume,
                sagittal_pdmap_volume,
                coronal_pdmap_volume
            ]
          
            median_t1map = np.median(t1maps, axis=0)
            median_pdmap = np.median(pdmaps, axis=0)

            # Save each as a NIfTI file
            median_nifti_t1map = nib.Nifti1Image(median_t1map, affine, header)
            nib.save(median_nifti_t1map, Path(output_path) / f'{filename}_t1map.nii.gz')

            median_nifti_pdmap = nib.Nifti1Image(median_pdmap, affine, header)
            nib.save(median_nifti_pdmap, Path(output_path) / f'{filename}_pdmap.nii.gz')
            
            print(f"T1 and PD maps saved to {output_path}.")

            # Calculate Multi-TI image
            multi_ti_calc = MultiTICalc()

            median_t1map_tensor = torch.from_numpy(median_t1map).float()
            median_pdmap_tensor = torch.from_numpy(median_pdmap).float()

            ti_tensor = torch.tensor(ti, dtype=torch.float32)
            tr_tensor = torch.tensor(4000, dtype=torch.float32)  # or whatever TR you desire

            ti_img_tensor = multi_ti_calc(median_t1map_tensor, median_pdmap_tensor, ti_tensor, tr_tensor)
            ti_img = ti_img_tensor.squeeze().cpu().numpy()

            # Save as NIfTI
            ti_img_nifti = nib.Nifti1Image(ti_img, affine, header)
            nib.save(ti_img_nifti, Path(output_path) / f'{filename}_{str(ti)}.nii.gz')

            print(f"Synthetic TI map saved to {Path(output_path) / f'{filename}_{str(ti)}.nii.gz'}")

def main():
    parser = argparse.ArgumentParser(description="Test the trained model on 3D T1w images.")
    parser.add_argument("--t1", type=str, help="Input path to 3D T1w image volume")
    parser.add_argument("--t2", type=str, help="Input path to 3D T2w image volume")
    parser.add_argument("--flair", type=str, help="Input path to 3D FLAIR image volume")
    parser.add_argument("--ti", type=int, help="Desired TI for synthetic image")
    parser.add_argument("--o", type=str, help="Output path for multi-TI image")
    parser.add_argument("--weights", type=str, help="Path to model weights")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to use (default: 0)")

    args = parser.parse_args()

    run_testing(args.t1, args.t2, args.flair, args.ti, args.o, args.weights, args.gpu)

if __name__ == "__main__":
    main()
