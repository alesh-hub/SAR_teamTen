import os
from PIL import Image
import numpy as np
from collections import defaultdict

def compute_class_pixel_counts(mask_paths, num_classes):
    """
    Compute the total pixel count for each class in the masks.
    
    Args:
        mask_paths (list): List of file paths to the mask images.
        num_classes (int): Total number of classes (e.g., 5 for {0, 1, 2, 3, 4}).
    
    Returns:
        dict: Dictionary with class IDs as keys and pixel counts as values.
        dict: Dictionary with class IDs as keys and normalized class weights as values.
    """
    class_pixel_counts = defaultdict(int)
    total_pixels = 0

    for mask_path in mask_paths:
        # Open mask image
        mask = Image.open(mask_path)
        mask_array = np.array(mask)

        # Ensure the mask is single-channel
        if mask_array.ndim != 2:
            raise ValueError(f"Mask {mask_path} is not a single-channel image.")

        # Count pixels per class
        for class_id in range(num_classes):
            class_pixel_counts[class_id] += np.sum(mask_array == class_id)

        # Update total pixel count
        total_pixels += mask_array.size

    # Compute normalized weights: inverse frequency
    class_weights = {
        class_id: total_pixels / (count + 1e-8)  # Add small epsilon to avoid division by zero
        for class_id, count in class_pixel_counts.items()
    }

    return class_pixel_counts, class_weights

# Usage
mask_dir = r'C:\Users\ale\Documents\GitHub\SAR_teamTen\dataset\train\labels_1D'
mask_paths = [os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir) if fname.endswith('.png')]

# Number of classes in the dataset
num_classes = 5  # Classes: {0, 1, 2, 3, 4}

# Compute pixel counts and weights
class_pixel_counts, class_weights = compute_class_pixel_counts(mask_paths, num_classes)

# Print results
print("Class Pixel Counts:")
for class_id, count in class_pixel_counts.items():
    print(f"Class {class_id}: {count} pixels")

print("\nClass Weights:")
for class_id, weight in class_weights.items():
    print(f"Class {class_id}: {weight:.6f}")
