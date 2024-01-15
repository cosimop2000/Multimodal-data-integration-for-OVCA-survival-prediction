import os
import h5py
import json

# Load configuration from JSON file
config_file_path = 'config.json'

if not os.path.isfile(config_file_path):
    raise FileNotFoundError(f"Configuration file '{config_file_path}' not found.")

with open(config_file_path, 'r') as config_file:
    config = json.load(config_file)

# Extract the openslide path
openslide_path = config.get('openslide_path', '')

# Check if the openslide path is provided
if not openslide_path:
    raise ValueError("Openslide path is not specified in the configuration file.")

# Use the openslide path
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(openslide_path):
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
    print(svs_path)
    slide = OpenSlide(svs_path)

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)
    i = 0
    for index, (x, y) in enumerate(patch_coords):
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


def process_directories(h5_directory, svs_directory, output_base_directory, patch_size=256):
    # Iterate through all .h5 files in the h5_directory
    for h5_file_name in os.listdir(h5_directory):
        if h5_file_name.endswith(".h5"):
            # Form full paths for the .h5 file and corresponding .svs file
            h5_file_path = os.path.join(h5_directory, h5_file_name)
            svs_file_name = os.path.splitext(h5_file_name)[0] + ".svs"
            svs_path = os.path.join(svs_directory, svs_file_name)

            # Create a directory based on the .h5 file name in the output base directory
            output_directory = os.path.join(output_base_directory, os.path.splitext(h5_file_name)[0])

            # Read patch coordinates from the .h5 file
            patch_coords = read_patch_coords_from_h5(h5_file_path)

            # Extract and save patches to the created directory
            extract_and_save_patches(svs_path, patch_coords, patch_size, output_directory)


# Example usage:
config_file_path = 'config.json'

if not os.path.isfile(config_file_path):
    raise FileNotFoundError(f"Configuration file '{config_file_path}' not found.")

with open(config_file_path, 'r') as config_file:
    config = json.load(config_file)

h5_directory = config.get('h5_directory', '')
svs_directory = config.get('svs_directory', '')
output_base_directory = config.get('output_base_directory', '')

if not h5_directory or not svs_directory or not output_base_directory:
    raise ValueError("h5_directory, svs_directory, and output_base_directory must be specified in the configuration file.")
process_directories(h5_directory, svs_directory, output_base_directory)
