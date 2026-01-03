import argparse
import os
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Import your previously defined classes here
from utils import UNet, MRI_Dataset, Run_UNet, VolumeAssembler

# Function to load the dataset and perform the test operation
def run_testing(input_path: str, output_path: str, pretrained_model: str, gpu: int = 0):
    # Setup the model
    model = UNet(in_ch=1, out_ch=1).to(f'cuda:{gpu}')
    checkpoint = torch.load(pretrained_model)  # Load your trained model
    model.load_state_dict(checkpoint['unet'])

    # Your input and output paths
    DataPath = Path(input_path)
    ResultPath = Path(output_path)

    # Function to load dataset
    test_loader = DataLoader(
        dataset=MRI_Dataset(DataPath, 'test'),
        batch_size=32,  # Example batch size
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    volume_assembler = VolumeAssembler(slice_shape=(192, 224), volume_shape=(192, 192, 224))  # Set appropriate shapes

    # Prepare to collect volumes for each orientation
    sagittal_volumes = []
    coronal_volumes = []
    axial_volumes = []
    
    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        for batch_id, (imgs, labels) in tqdm(enumerate(test_loader), total=len(test_loader)):
            imgs = imgs.to(f'cuda:{gpu}')
            outputs = model(imgs)

            # Assuming outputs are 2D slices; here you will need to handle orientation logic
            output_slices = [outputs[0, channel, :, :].cpu().numpy() for channel in range(outputs.shape[1])]
            
            # Assemble 3D volumes based on slice collection method you have (adapt as necessary)
            # Example:
            # - You can organize slices into orientations as needed
            if your_condition_for_sagittal:  # replace with your actual condition for orientation
                volume = volume_assembler.assemble_volume(output_slices)
                sagittal_volumes.append(volume)
            elif your_condition_for_coronal:  # replace with your actual condition for orientation
                volume = volume_assembler.assemble_volume(output_slices)
                coronal_volumes.append(volume)
            elif your_condition_for_axial:  # replace with your actual condition for orientation
                volume = volume_assembler.assemble_volume(output_slices)
                axial_volumes.append(volume)

    # Compute median volumes from collected volumes for each orientation
    if sagittal_volumes:
        median_sagittal = volume_assembler.compute_median_volume(sagittal_volumes)
        np.save(os.path.join(ResultPath, 'median_sagittal.npy'), median_sagittal)  # Save/handle medial volume as needed

    if coronal_volumes:
        median_coronal = volume_assembler.compute_median_volume(coronal_volumes)
        np.save(os.path.join(ResultPath, 'median_coronal.npy'), median_coronal)  # Save/handle medial volume as needed

    if axial_volumes:
        median_axial = volume_assembler.compute_median_volume(axial_volumes)
        np.save(os.path.join(ResultPath, 'median_axial.npy'), median_axial)  # Save/handle medial volume as needed
        
def main():
    parser = argparse.ArgumentParser(description="Test the trained model on 3D T1w images.")
    parser.add_argument("input_path", type=str, help="Input path to 3D T1w image volume")
    parser.add_argument("output_path", type=str, help="Output path for multi-TI image")
    parser.add_argument("pretrained_model", type=str, help="Path to the pretrained model weights")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to use (default: 0)")

    args = parser.parse_args()

    run_testing(args.input_path, args.output_path, args.pretrained_model, args.gpu)

if __name__ == "__main__":
    main()
