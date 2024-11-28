# TO DO LIST

- create notebook 'showroom.ipynb' to display custom transformations

- create code lines to zip the best checkpoint at the end of training and then unzip during the evaluation mode.


## PAPER REPLICATION

- transformations:
    - resize 320 x 320
    - random horizontal flip and/or random vertical flip
    ```python
    def augment_images(dataset_images):
    augmented_images = []

    for image in dataset_images:
        # Convert image to a PyTorch tensor if it's not already
        if not isinstance(image, torch.Tensor):
            image = T.ToTensor()(image)
        
        # Randomly decide whether to apply augmentation (50% chance)
        if random.random() < 0.5:
            # Step 1: Random flipping
            flip_horizontal = random.random() < 0.5
            flip_vertical = random.random() < 0.5

            # Ensure at least one flip is applied
            if not (flip_horizontal or flip_vertical):
                if random.random() < 0.5:
                    flip_horizontal = True
                else:
                    flip_vertical = True

            # Apply the flips
            if flip_horizontal:
                image = F.hflip(image)
            if flip_vertical:
                image = F.vflip(image)

            augmented_images.append(image)
        else:
            # Keep the original image if no augmentation is applied
            augmented_images.append(image)
    
    return augmented_images

    ``` 
    - setting the crop to be random (p = 0.0)

## To do in the repo:


- create different folders for the training of each model, containing:
    - train/test file for paper preprocessing
    - train/test file for custom preprocessing
