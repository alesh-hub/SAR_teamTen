# General libraries
import os
import random
import sys

import numpy as np

# Get the directory containing the current file and add it to Python's search path
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory)

import sys

import torch
# Image processing
from PIL import Image
from pytorch_lightning import LightningDataModule
# Scikit-learn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
# PyTorch Vision
from torchvision import transforms
# SAR Tranformations
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

print(sys.modules)
class MaskToTensor:
    def __call__(self, mask):
        # Converts mask to a PyTorch tensor with dtype=torch.int64
        return torch.as_tensor(np.array(mask), dtype=torch.int64)


# Class to operate image and mask random flips
class RandomizedFlip:
    def __init__(self, flip_probability=0.5):
        """
        Args:
            flip_probability (float): Probability of applying a flip (default: 0.5).
        """
        self.flip_probability = flip_probability
        self.flip_type = "none"  # Initialize flip type

    def __call__(self, image, mask):
        """
        Randomly applies a horizontal flip, vertical flip, or both to the image and mask.

        Args:
            image (PIL Image): The input image.
            mask (PIL Image): The corresponding mask.

        Returns:
            (PIL Image, PIL Image): The flipped image and mask.
        """
        if random.random() < self.flip_probability:
            # Select the flip type and store it
            self.flip_type = random.choices(
                ['horizontal', 'vertical', 'both'], weights=[1/3, 1/3, 1/3], k=1
            )[0]
            
            if self.flip_type == 'horizontal':
                image = F.hflip(image)
                mask = F.hflip(mask)
            elif self.flip_type == 'vertical':
                image = F.vflip(image)
                mask = F.vflip(mask)
            elif self.flip_type == 'both':
                image = F.hflip(F.vflip(image))
                mask = F.hflip(F.vflip(mask))

        else:
            self.flip_type = "none"  # Set flip type to none if no flip is applied
            
            
        return image, mask

# Class to operate image and mask resizing
class RandomizedResize:
    def __init__(self, scale_range=(0.5, 1.5)):
        """
        Args:
            scale_range (tuple): A tuple specifying the min and max scale factors (default: (0.5, 1.5)).
        """
        self.scale_range = scale_range

    def __call__(self, image, mask):
        """
        Randomly resizes the image and mask within the specified scale range.

        Args:
            image (PIL Image): The input image.
            mask (PIL Image): The corresponding mask.

        Returns:
            (PIL Image, PIL Image): The resized image and mask.
        """
        # Generate a random scale factor within the scale range
        scale_factor = random.uniform(*self.scale_range)
        print(scale_factor)
        
        print("Original Image Size: ", image.size)
        print("Origina Mask Size: ", mask.size)

        # Compute new dimensions
        new_width = int(image.width * scale_factor)
        new_height = int(image.height * scale_factor)
        print("Resizing by a factor of ", scale_factor)


        # Resize both the image and the mask
        image = F.resize(image, (new_height, new_width), interpolation=F.InterpolationMode.BILINEAR)
        mask = F.resize(mask, (new_height, new_width), interpolation=F.InterpolationMode.NEAREST)
        print("New Image Size: ", image.size)
        print("New Mask Size: ", mask.size)
        return image, mask

class RandomizedCrop:
    def __init__(self, crop_size=320):
        """
        Args:
            crop_transform (callable): The focused crop transformation.
            crop_size (int): The size of the random crop.
            probability (float): The probability of applying the focused crop.
        """
        self.crop_size = crop_size

    def __call__(self, image, mask):
        """
        Applies the focused crop transformation with a given probability if an oil spill is present.
        Otherwise, performs a random crop of the specified size.
        """

        # Otherwise, perform a random crop
        width, height = image.size
        top = random.randint(0, max(0, height - self.crop_size))
        left = random.randint(0, max(0, width - self.crop_size))

        # Perform random crop on both image and mask
        cropped_image = F.crop(image, top, left, self.crop_size, self.crop_size)
        cropped_mask = F.crop(mask, top, left, self.crop_size, self.crop_size)

        return cropped_image, cropped_mask


class SARDataModule(LightningDataModule):
    
    def __init__(self, data_dir: str = "./", batch_size: int = 8, val_split: float = 0.2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split

        # Transformation for images
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[0:1, :, :])  # Keep only the first channel (C=1, H, W)

        ])

        # Transformation for masks
        self.mask_transform = transforms.Compose([
            MaskToTensor()
        ])
        
        # Joint transformation for both image and mask
        self.joint_transform = transforms.Compose([
            RandomizedResize(),
            RandomizedFlip(),
            RandomizedCrop()
        ])
        
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
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=16, shuffle=True, persistent_workers=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, num_workers=16, batch_size=self.batch_size, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, num_workers=16, batch_size=self.batch_size, persistent_workers=True)

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
                mask = MaskToTensor(mask)
            

            mask = mask.squeeze(0).long()

        # Return both the image and its corresponding mask
        return img, mask