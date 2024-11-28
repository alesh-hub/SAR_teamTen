import torch
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

# Instantiate the model
model = deeplabv3_mobilenet_v3_large(pretrained=True)
model.eval()

# Function to test different input sizes
def test_input_size(height, width):
    try:
        # Create a dummy input tensor with the specified size
        input_tensor = torch.randn(1, 3, height, width)
        # Pass the tensor through the model
        output = model(input_tensor)
        print(f"Input size ({height}, {width}) is supported.")
    except Exception as e:
        print(f"Input size ({height}, {width}) is not supported. Error: {e}")

# Test various input sizes
test_input_size(224, 224)  # Common size, should work
test_input_size(320, 320)  # Multiple of 32, should work
test_input_size(250, 250)  # Not a multiple of 32, may cause issues
