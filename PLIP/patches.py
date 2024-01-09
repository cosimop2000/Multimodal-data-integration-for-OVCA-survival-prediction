import h5py
from PIL import Image
import numpy as np

# Step 1: Load Patches from HDF5 File
with h5py.File('C:\\Users\\pavon\\Documents\\Bioinformatics_project\\Data\\WSI\\RESULTS_DIRECTORY\\patches\\'
               'TCGA-25-1878-01A-01-TS1.d6219de6-280f-4dd9-ab5d-4598c4781b14.h5', 'r') as h5_file:

    patches = h5_file['coords'][:]


# Step 2: Reconstruct Image from Patches
patch_size = int(np.sqrt(patches.shape[0]))
num_patches = int(np.sqrt(patches.shape[1]))

print(patches.shape)
print(patches[1])
print(patches)

"""
# Reshape patches to form a grid
reconstructed_image = patches.reshape((num_patches, num_patches, patch_size, patch_size))

# Combine patches into the final image
reconstructed_image = reconstructed_image.transpose(0, 2, 1, 3).reshape(num_patches * patch_size, num_patches * patch_size)

# Step 3: Normalize if needed

# Step 4: Resize if needed
# reconstructed_image = ...

# Step 5: Save or Use the Reconstructed Image
Image.fromarray(reconstructed_image.astype(np.uint8)).save('reconstructed_image.jpg')
"""

