import os
from PIL import Image
import numpy as np

def get_unique_values_from_masks(label_dir, max_images=30):
    """
    Function to get unique pixel values from single-channel masks 
    in the first `max_images` images in the directory.

    Args:
        label_dir (str): Path to the directory containing label images.
        max_images (int): Maximum number of images to process.

    Returns:
        set: A set of unique pixel values found in the masks.
    """
    unique_values = set()

    # Iterate over the first `max_images` files in the directory
    for idx, file_name in enumerate(sorted(os.listdir(label_dir))):
        if idx >= max_images:  # Stop after processing max_images
            break

        file_path = os.path.join(label_dir, file_name)

        # Check if the file is an image
        if file_name.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            print(f"Processing: {file_name}")
            
            # Open the image
            with Image.open(file_path) as img:
                # Ensure the mask is single-channel (convert if necessary)
                img_np = np.array(img)
                if len(img_np.shape) > 2:  # Convert RGB to single-channel if needed
                    raise ValueError(f"Image {file_name} is not a single-channel mask.")

                # Get unique values in the image and add to the set
                unique_values.update(np.unique(img_np))

    return unique_values


if __name__ == "__main__":
    # Specify the directory containing label images
    label_dir = r"C:\Users\ale\Documents\GitHub\SAR_teamTen\dataset\train\labels_1D"  # Update path as needed

    # Get unique values from the first 30 images
    unique_values = get_unique_values_from_masks(label_dir, max_images=30)

    # Print the unique values
    print("\nUnique pixel values in the first 30 label images:")
    for value in sorted(unique_values):
        print(value)
