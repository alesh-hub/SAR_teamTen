# General libraries
import os
import sys
import numpy as np
# Get the directory containing the current file and add it to Python's search path
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory)

# stats for dataset normalization (train set): 
# Computed mean: 0.5185160303047751, std: 0.24470906979034782

# Image processing
from PIL import Image
from pytorch_lightning import LightningDataModule

# Scikit-learn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch
# PyTorch Vision
from torchvision import transforms
# SAR Tranformations
import custom_transforms
from torchvision.transforms import InterpolationMode

class SARDataModule(LightningDataModule):
    
    def __init__(self, data_dir: str = "./", batch_size: int = 8, val_split: float = 0.2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split

        # Transformation for images
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5185160303047751]*3, std=[0.24470906979034782]*3)
            # transforms.Resize((320, 320), interpolation=InterpolationMode.BILINEAR)
            # transforms.Lambda(lambda x: x[0, :, :].unsqueeze(0))  # Take the first channel
            # Augmentation TO DO
            # TO ADD ProbabilisticRandomizedFocusedCrop
        ])

        # Transformation for masks
        self.mask_transform = transforms.Compose([
            custom_transforms.MaskToTensor(),
            # transforms.Resize((320, 320), interpolation=InterpolationMode.NEAREST)
            # No channel selection for masks
        ])
        
        # Joint transformation for both image and mask
        self.joint_transform = custom_transforms.ProbabilisticRandomizedFocusedCrop(
            crop_transform=custom_transforms.RandomizedFocusedCrop(crop_size=320, max_shift=80),
            crop_size=320,
            probability=0.4
        )
        
    def prepare_data(self) -> None:
        # Not needed in our case, no download or labelling needed
        # prepare_data(self) is used for operations that run only once and on one process.
        pass 

        
    def setup(self, stage: str = None) -> None:
        
        # Helper function to get sorted file names
        def get_sorted_file_names(folder_path):
            return sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

        # ----------------- TRAIN ----------------- 
        # Paths to the images and masks directories
        train_images_dir = os.path.join(self.data_dir, 'train/images')
        train_masks_dir = os.path.join(self.data_dir, 'train/labels_1D')

        # Get the list of image filenames
        train_images_filenames = get_sorted_file_names(train_images_dir)

        # Generate full paths for images and masks
        train_images_paths = [os.path.join(train_images_dir, f) for f in train_images_filenames]
        train_masks_paths = [os.path.join(train_masks_dir, os.path.splitext(f)[0] + '.png') for f in train_images_filenames]

        # Split into train and validation sets
        train_images_paths, val_images_paths, train_masks_paths, val_masks_paths = train_test_split(
            train_images_paths, train_masks_paths, test_size=self.val_split, random_state=42)
        
        # ----------------- TEST ----------------- 
        # Paths to the test dataset
        test_images_dir = os.path.join(self.data_dir, 'test/images')
        test_masks_dir = os.path.join(self.data_dir, 'test/labels_1D')

        # Get the list of test image filenames
        test_images_filenames = get_sorted_file_names(test_images_dir)

        # Generate full paths for test images and masks
        test_images_paths = [os.path.join(test_images_dir, f) for f in test_images_filenames]
        test_masks_paths = [os.path.join(test_masks_dir, os.path.splitext(f)[0] + '.png') for f in test_images_filenames]

        # ----------------- LOADING ----------------- 
        if stage == "fit" or stage is None:
            self.train_dataset = SARImageDataset(
                train_images_paths, train_masks_paths,
                image_transform=self.image_transform, mask_transform=self.mask_transform, 
                joint_transform=self.joint_transform
            )
            self.val_dataset = SARImageDataset(
                val_images_paths, val_masks_paths,
                image_transform=self.image_transform, mask_transform=self.mask_transform,
                joint_transform=None
            )
        if stage == "test" or stage is None:
            self.test_dataset = SARImageDataset(
                test_images_paths, test_masks_paths,
                image_transform=self.image_transform, mask_transform=self.mask_transform,
                joint_transform=None
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8 ,shuffle=True, persistent_workers=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, num_workers=8 , batch_size=self.batch_size, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, num_workers=8, batch_size=self.batch_size, persistent_workers=True)

class SARImageDataset(Dataset): # Allows the user to apply a custom transformation via self.mask_transform. 

    def __init__(self, images_paths, masks_paths, image_transform=None, 
                 mask_transform=None, joint_transform=None):
        self.images_paths = images_paths
        self.masks_paths = masks_paths
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.joint_transform = joint_transform
        
    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        # Load the image
        image_path = self.images_paths[idx]
        
        # Load the corresponding mask
        mask_path = self.masks_paths[idx]
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        
        # Open images
        with Image.open(image_path) as img, Image.open(mask_path) as mask:
            # Apply joint transformation if available
            if self.joint_transform:
                img, mask = self.joint_transform(img, mask)

            # Apply individual transformations
            if self.image_transform:
                img = self.image_transform(img)
            else:
                img = transforms.ToTensor()(img)
            
            # Apply specified transformations and convert to tensor
            if self.mask_transform:
                mask = self.mask_transform(mask)
            else: 
                mask = transforms.ToTensor()(mask)
            

            mask = mask.squeeze(0).long()

        # Return both the image and its corresponding mask
        return img, mask