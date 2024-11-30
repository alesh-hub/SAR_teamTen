import os
from PIL import Image
import numpy as np

def get_unique_colors(label_dir):
    """
    Function to get unique colors from all label images in the directory.
    
    Args:
        label_dir (str): Path to the directory containing label images.
        
    Returns:
        set: A set of unique colors found in all label images.
    """
    unique_colors = set()

    # Iterate over all files in the directory
    for file_name in sorted(os.listdir(label_dir)):
        file_path = os.path.join(label_dir, file_name)
        
        # Check if the file is an image
        if file_name.endswith(('.png')):
            #print(f"Processing: {file_name}")
            
            # Open the image
            with Image.open(file_path) as img:
                img_np = np.array(img)  # Convert image to numpy array
                
                # Get unique colors in the image and add to the set
                colors = {tuple(color) for color in img_np.reshape(-1, img_np.shape[-1])}
                unique_colors.update(colors)

    return unique_colors


if __name__ == "__main__":
    # Specify the directory containing label images
    label_dir = r"C:\Users\ale\Documents\GitHub\SAR_teamTen\dataset\train\labels"  # Update path as needed

    # Get unique colors
    unique_colors = get_unique_colors(label_dir)

    # Print the unique colors
    print("\nUnique colors in label images:")
    for color in unique_colors:
        print(color)
