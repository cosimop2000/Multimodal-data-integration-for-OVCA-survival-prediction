import torch

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available! Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

# Create a tensor on the selected device
x = torch.randn(3, 3).to(device)

# Print the tensor and the device it is on
print("Tensor:")
print(x)
print("\nDevice:")
print(x.device)
