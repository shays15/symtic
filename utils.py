from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
import torchvision.models as models

import torch.nn as nn
import torch.nn.functional as F
import torchvision  # modules and transforms for computer vision
from tqdm.auto import tqdm  # progress bar
import os
from glob import glob
import nibabel as nib
import nibabel.processing

class MRI_Dataset(Dataset):
    def __init__(self, dir, mode='train'):
        self.mode = mode
        self.data_path = Path(dir)
        (self.mprage_paths, self.t2w_paths, self.flair_paths, self.fgaitr_paths) = self._get_paths()

        # Automatically calculate the number of slices based on the first image
        sample_img = nib.load(str(self.mprage_paths[0]))

    def _get_paths(self):
        modalities = ["mprage", "t2w", "flair", "fgatir"]
        
        mprage_paths, t2w_paths, flair_paths, fgatir_paths = [], [], [], []
    
        for modality in modalities:
            # Get path to the current modality
            modality_dir = Path(self.data_path) / self.mode / modality
            
            # Check if the directory exists, if not skip this modality
            if not modality_dir.exists():
                print(f"Directory {modality_dir} does not exist!")
                continue
            
            # Collect and filter image paths for the current modality
            for img_path in sorted(modality_dir.glob("*")):
                # Remove the '.nii.gz' extension and extract the slice number
                slice_number_str = img_path.stem
                if img_path.suffixes[-1] == ".gz":
                    slice_number_str = slice_number_str.rsplit('.', 1)[0]
                slice_number = int(slice_number_str.split('_')[-1])
                orientation = slice_number_str.rsplit('_')[-3]
                #print(orientation)

                # Only include slices in the range of 60-130
                #if slice_number==100: #overfit
                #if 60 <= slice_number <= 130: #training
                if 0 <= slice_number <= 224 and orientation == 'sagittal': #testing
                    if modality == "mprage":
                        mprage_paths.append(img_path)
                    elif modality == "t2w":
                        t2w_paths.append(img_path)
                    elif modality == "flair":
                        flair_paths.append(img_path)
                    elif modality == "fgatir":
                        fgatir_paths.append(img_path)

        return mprage_paths, t2w_paths, flair_paths, fgatir_paths
        
    def __len__(self):
        return len(self.mprage_paths)

    def normalize(self, img):
        max_val = np.percentile(img, 99.5)
        min_val = 0
        if max_val == min_val:
            # Return a normalized image of zeros (or choose another behavior)
            return np.zeros_like(img)
        img = (img - min_val) / (max_val - min_val)
        return img

    def __getitem__(self, idx):
            
        mprage_img = nib.load(str(self.mprage_paths[idx])).get_fdata().astype(np.float32)
        mprage_img = self.normalize(mprage_img)
        
        t2w_img = nib.load(str(self.t2w_paths[idx])).get_fdata().astype(np.float32)
        t2w_img = self.normalize(t2w_img)
        
        flair_img = nib.load(str(self.flair_paths[idx])).get_fdata().astype(np.float32)
        flair_img = self.normalize(flair_img)        
        
        fgatir_img = nib.load(str(self.fgatir_paths[idx])).get_fdata().astype(np.float32)
        fgatir_img = self.normalize(fgatir_img)      
        
        mprage_2d = torch.tensor(mprage_img, dtype=torch.float32)
        t2w_2d = torch.tensor(t2w_img, dtype=torch.float32)
        flair_2d = torch.tensor(flair_img, dtype=torch.float32)
        fgatir_2d = torch.tensor(fgatir_img, dtype=torch.float32)

        
        img = torch.stack([mprage_2d, t2w_2d, flair_2d], dim=0)
        label = torch.stack([fgatir_2d], dim=0)
        
        return img, label

class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, conditional_ch=0, num_lvs=4, base_ch=16, final_act='relu'):
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
        #up_sampled_tensor = F.interpolate(in_tensor, size=None, scale_factor=2, mode='bilinear', align_corners=False)
        #up_sampled_tensor = self.conv(up_sampled_tensor)
        #up_sampled_tensor = torch.cat([up_sampled_tensor, torch.zeros(up_sampled_tensor.shape[0], up_sampled_tensor.shape[1],up_sampled_tensor.shape[2], 1).to(torch.device("cuda:0"))], 3) #I added for error
        up_sampled_tensor = F.interpolate(in_tensor, size=None, scale_factor=2, mode='bilinear', align_corners=False)
        up_sampled_tensor = self.conv(up_sampled_tensor)
        return torch.cat([encoded_feature, up_sampled_tensor], dim=1)


        print(encoded_feature.shape)
        print(up_sampled_tensor.shape)
        return torch.cat([encoded_feature, up_sampled_tensor], dim=1)
      
class Run_UNet:

    def __init__(self, in_ch, out_ch, out_dir, lr, pretrained_model=None, gpu=0):
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.out_dir = out_dir
        self.device = torch.device("cuda:0" if gpu==0 else "cuda:1")

        self.L1_loss = nn.L1Loss().to(self.device)
        
        #vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.to(self.device)
        #self.perceptual_loss = PerceptualLoss(vgg)
        
        #self.bce_loss = FocalLoss(gamma=2.0)
        #self.L2_loss = nn.MSELoss().to(self.device)

        self.unet = UNet(in_ch=in_ch, out_ch=out_ch)
        self.optim_unet = torch.optim.Adam(self.unet.parameters(), lr=lr)
        self.start_epoch = 0
        self.unet.to(self.device)
        self.checkpoint = None
        
        if pretrained_model is not None:
            self.checkpoint = torch.load(pretrained_model, map_location=self.device)
            self.unet.load_state_dict(self.checkpoint['unet'])
            self.optim_unet.load_state_dict(self.checkpoint['optim_unet'])
            self.start_epoch = self.checkpoint['epoch']
        
        self.start_epoch += 1
        self.loss_store=[]
        mkdir_p(os.path.join(out_dir, 'results'))
        mkdir_p(os.path.join(out_dir, 'models'))
        mkdir_p(os.path.join(out_dir, 'logs'))
        self.writer = SummaryWriter(os.path.join(out_dir, 'logs'))

        self.header = load_nifti_header('/iacl/pg23/savannah/data/umdctmri/UMDCTMRI-008/00/proc/UMDCTMRI-008_00_00-31_BRAIN-T1-MPRAGE-3D-SAGITTAL-PRE_hdrfix_n4_reg.nii.gz')
        self.affine = load_nifti_affine('/iacl/pg23/savannah/data/umdctmri/UMDCTMRI-008/00/proc/UMDCTMRI-008_00_00-31_BRAIN-T1-MPRAGE-3D-SAGITTAL-PRE_hdrfix_n4_reg.nii.gz')


    def load_dataset(self, dir, batch_size):

        self.train_loader = torch.utils.data.DataLoader(
            dataset=MRI_Dataset(DataPath, 'train'),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            dataset=MRI_Dataset(DataPath, 'val'),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            dataset=MRI_Dataset(DataPath, 'test'),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
    def train(self, epochs):
        best_val_loss = float('inf')  # Initialize the best loss to infinity
        tensorboard_index=0
        for epoch in range(self.start_epoch, epochs+1):
            #self.train_loader = tqdm(self.train_loader)
            self.unet.train()
            epoch_loss = 0.0
            
            progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{epochs}', total=len(self.train_loader))

            for batch_id, (imgs, labels) in enumerate(progress_bar):
                
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                self.optim_unet.zero_grad()

                outputs = self.unet(imgs)

                loss = self.L1_loss(outputs, labels)

                loss.backward()
                self.optim_unet.step()
                
                epoch_loss += loss.item()
                
                progress_bar.set_postfix({'Total Loss': '{:.4f}'.format(loss.item())})

                if batch_id % 10 == 0 and epoch % 10 == 0:
                    self.save_3d_volume(imgs, self.affine, self.header, epoch, batch_id, 'source','train', is_ct=False)
                    self.save_3d_volume(outputs, self.affine, self.header, epoch, batch_id, 'pred','train', is_ct=True)
                    self.save_3d_volume(labels, self.affine, self.header, epoch, batch_id, 'gt','train', is_ct=True)

            epoch_loss /= len(self.train_loader)

            print(f'Epoch {epoch}/{epochs}, Total loss:{epoch_loss:.4f}')
            self.writer.add_scalar('Loss/train_epoch_avg', epoch_loss, epoch)

            val_loss = self.validate(epoch)

            self.save_model(os.path.join(self.out_dir, 'models', f'unet_epoch{epoch}.pth'), epoch, epoch_loss)
        
            # Save model after each epoch
            if val_loss < best_val_loss:
                print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
                best_val_loss = val_loss
                self.save_model(os.path.join(self.out_dir, 'models', f'best_unet_epoch{epoch}.pth'), epoch, epoch_loss)
                
        self.writer.close()
                
    def validate(self, epoch):
        self.unet.eval()  # Set the model to evaluation mode
        self.val_loader = tqdm(self.val_loader)
        epoch_loss = 0.0
        loss = 0.0
        
        with torch.no_grad():  # Disable gradient calculation
            for batch_id, (imgs, labels) in enumerate(self.val_loader):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.unet(imgs)
                loss = self.L1_loss(outputs, labels)
                epoch_loss += loss.item()

                # Optionally save validation results here
                if batch_id % 5 == 0 and epoch % 10 == 0:
                    self.save_3d_volume(imgs, self.affine, self.header, epoch, batch_id, 'source','val', is_ct=False)
                    self.save_3d_volume(outputs, self.affine, self.header, epoch, batch_id, 'pred','val', is_ct=True)
                    self.save_3d_volume(labels, self.affine, self.header, epoch, batch_id, 'gt','val', is_ct=True)
        
        epoch_loss /= len(self.val_loader)

        print(f'Validation Loss: {epoch_loss:.4f}')
        self.writer.add_scalar('Loss/val', epoch_loss, epoch)
        return epoch_loss

    def test(self, epoch):
        # Testing
        self.test_loader = tqdm(self.test_loader)
        self.unet.eval()

        test_loss_sum = 0.0
        
        with torch.set_grad_enabled(False):
            
            for batch_id, (imgs,labels) in enumerate(self.test_loader):
            #for batch_id, (imgs) in enumerate(self.test_loader):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                #imgs = imgs.to(self.device)
                outputs = self.unet(imgs)
                loss = self.L1_loss(outputs,labels)

                # save test results
                self.save_3d_volume(imgs, self.affine, self.header, epoch, batch_id, 'source','test', is_ct=False)
                self.save_3d_volume(outputs, self.affine, self.header, epoch, batch_id, 'pred','test', is_ct=True)
                self.save_3d_volume(labels, self.affine, self.header, epoch, batch_id, 'gt','test', is_ct=True)

                
    def save_3d_volume(self, volume_data, affine, header, epoch, batch_id, prefix, phase, is_ct):
        results_dir = os.path.join(self.out_dir, 'results', phase)
        mkdir_p(results_dir)
        
        # It is [32, 1, 192, 224]
        volume_data = volume_data.detach()

        # If this is a CT image, apply unnormalization
        if is_ct:
            volume_data = volume_data * 5000 - 1000  # Reverse normalization

        # Split the channels and concatenate along the depth dimension
        for channel in range(volume_data.shape[1]):
            # Extract the data for the current channel
            channel_data = volume_data[:, channel, ...]
    
            # channel_data is now of shape [32, 192, 224], which we need to transpose to [192, 224, 32]
            # to match the desired shape for saving as a NIfTI volume
            channel_data = channel_data.permute(1, 2, 0).contiguous()
    
            # The contiguous() call is necessary if you're going to use numpy() on the tensor later,
            # as the permute might not actually change the memory layout of the tensor, only the view.
    
            # Construct the file name for the current channel
            file_name = os.path.join(results_dir,f'{prefix}_epoch_{epoch}_batch_{batch_id}_channel_{channel}.nii.gz')
    
            # Save the nifti image
            nifti_img = nib.Nifti1Image(channel_data.cpu().numpy(), affine, header)
            nib.save(nifti_img, file_name)
    
            #print(f'Saved {prefix} channel {channel} volume from epoch {epoch} batch {batch_id} in {phase} to {file_name}')

    def save_model(self, file_name, epoch, epoch_loss):
        state = {'epoch' : epoch,
                 'loss': epoch_loss,
                 'unet' : self.unet.state_dict(),
                 'optim_unet' : self.optim_unet.state_dict()}
        torch.save(obj=state, f=file_name)
        print(f'Model saved to ==> {file_name}')
      
class VolumeAssembler:
    def __init__(self, slice_shape, volume_shape):
        # Assuming slice_shape is (H, W) for the 2D slices
        self.slice_shape = slice_shape  # Height, Width of a single slice
        self.volume_shape = volume_shape  # (D, H, W) for the 3D volume

    def assemble_volume(self, output_slices):
        """
        Assembles 2D slices into a 3D volume.
        :param output_slices: A list of 2D slices (numpy arrays).
        :return: A 3D volume (numpy array).
        """
        volume = np.zeros(self.volume_shape, dtype=np.float32)

        # Fill the volume with slices. 
        # Adjust the loop iterator as per your logic to fill slices correctly.
        for i, slice_ in enumerate(output_slices):
            if i < self.volume_shape[0]:
                volume[i, :, :] = slice_

        return volume
    
    def compute_median_volume(self, volumes):
        """
        Computes the median volume from a list of 3D volumes.
        :param volumes: List of 3D volumes (numpy arrays).
        :return: Median volume (numpy array).
        """
        return np.median(volumes, axis=0)
