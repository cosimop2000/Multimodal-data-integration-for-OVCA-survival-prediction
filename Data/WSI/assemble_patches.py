import os
import h5py

# The path can also be read from a config file, etc.
OPENSLIDE_PATH = r'C:\\Users\\pavon\Downloads\\openslide-win64-20231011\\openslide-win64-20231011\\bin'
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        from openslide import OpenSlide
else:
    from openslide import OpenSlide

from PIL import Image

def read_patch_coords_from_h5(h5_file_path):
    with h5py.File(h5_file_path, 'r') as h5_file:
        # Assuming the coordinates are stored as 'coords' dataset in the HDF5 file
        patch_coords = h5_file['coords'][:]
    return patch_coords

def extract_and_save_patches(svs_path, patch_coords, patch_size, output_directory):
    # Open the whole-slide image
    slide = OpenSlide(svs_path)

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)
    i = 0
    for index, (x, y) in enumerate(patch_coords):
        ++i
        if i == 3:
            break
        # Extract the patch from the whole-slide image
        patch = slide.read_region((x, y), 0, (patch_size, patch_size))

        # Convert the patch to RGB format
        patch = patch.convert("RGB")

        # Save the patch as PNG or JPG
        patch.save(os.path.join(output_directory, f"patch_{index}.png"), "PNG")
        # Alternatively, you can use the following line to save as JPG:
        # patch.save(os.path.join(output_directory, f"patch_{index}.jpg"), "JPEG")

    # Close the whole-slide image
    slide.close()

# Example usage:
svs_path = ("C:\\Users\\pavon\\Documents\\Bioinformatics_project\\"
            "Data\\WSI\\DATA_DIRECTORY\\TCGA-25-1878-01A-01-TS1.d6219de6-280f-4dd9-ab5d-4598c4781b14.svs")
h5_file_path = ("C:\\Users\\pavon\\Documents\\Bioinformatics_project\\Data\\WSI\\RESULTS_DIRECTORY\\"
                "patches\\TCGA-25-1878-01A-01-TS1.d6219de6-280f-4dd9-ab5d-4598c4781b14.h5")
patch_size = 256  # Adjust the patch size as needed
output_directory = "C:\\Users\\pavon\\Documents\\Bioinformatics_project\\PLIP\\images"

patch_coords = read_patch_coords_from_h5(h5_file_path)
extract_and_save_patches(svs_path, patch_coords, patch_size, output_directory)
