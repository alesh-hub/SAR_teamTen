import random
import numpy as np
from torchvision.transforms import functional as F
import torch

class ProbabilisticRandomizedFocusedCrop:
    def __init__(self, crop_transform, crop_size=320, probability=0.5):
        """
        Args:
            crop_transform (callable): The focused crop transformation.
            crop_size (int): The size of the random crop.
            probability (float): The probability of applying the focused crop.
        """
        self.crop_transform = crop_transform
        self.crop_size = crop_size
        self.probability = probability

    def __call__(self, image, mask):
        """
        Applies the focused crop transformation with a given probability if an oil spill is present.
        Otherwise, performs a random crop of the specified size.
        """
        # Convert mask to numpy to check for oil spill
        mask_np = np.array(mask)
        oil_spill_present = np.any(mask_np == 1)

        # Apply focused crop with the given probability if oil spill is present
        if oil_spill_present and random.random() < self.probability:
            cropped_image, cropped_mask = self.crop_transform(image, mask)
            return cropped_image, cropped_mask # Boolean only for visualization, TO BE REMOVED!

        # Otherwise, perform a random crop
        width, height = image.size
        top = random.randint(0, max(0, height - self.crop_size))
        left = random.randint(0, max(0, width - self.crop_size))

        # Perform random crop on both image and mask
        cropped_image = F.crop(image, top, left, self.crop_size, self.crop_size)
        cropped_mask = F.crop(mask, top, left, self.crop_size, self.crop_size)

        return cropped_image, cropped_mask
    
    
    


class RandomizedFocusedCrop:
    def __init__(self, crop_size=320, max_shift=20):
        """
        Args:
            crop_size (int): Size of the crop.
            max_shift (int): Maximum number of pixels to randomly shift the crop center.
        """
        self.crop_size = crop_size
        self.max_shift = max_shift

    def __call__(self, image, mask):
        # Convert mask to numpy
        mask_np = np.array(mask)
        # Focus only on oil spill label (1)
        oil_spill_mask = (mask_np == 1).astype(np.uint8)

        # Find oil spill region in mask
        oil_spill_indices = np.argwhere(oil_spill_mask > 0)

        if len(oil_spill_indices) > 0:
            # Calculate bounding box around oil spill
            y_min, x_min = oil_spill_indices.min(axis=0)
            y_max, x_max = oil_spill_indices.max(axis=0)

            # Calculate initial crop center
            center_y = (y_min + y_max) // 2
            center_x = (x_min + x_max) // 2

            # Apply random shift to the crop center
            center_y += random.randint(-self.max_shift, self.max_shift)
            center_x += random.randint(-self.max_shift, self.max_shift)

            # Determine crop box
            half_crop = self.crop_size // 2
            top = max(center_y - half_crop, 0)
            left = max(center_x - half_crop, 0)
            bottom = top + self.crop_size
            right = left + self.crop_size

            # Adjust if crop exceeds image boundaries
            if bottom > image.height:
                bottom = image.height
                top = bottom - self.crop_size
            if right > image.width:
                right = image.width
                left = right - self.crop_size

            # Crop the image and mask
            image = F.crop(image, top, left, self.crop_size, self.crop_size)
            mask = F.crop(mask, top, left, self.crop_size, self.crop_size)

        return image, mask
    
    
    
class MaskToTensor:
    def __call__(self, mask):
        # Converts mask to a PyTorch tensor with dtype=torch.int64
        return torch.as_tensor(np.array(mask), dtype=torch.int64)


