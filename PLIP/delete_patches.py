import os
import json

def delete_png_files_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg"):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                # print(f"Deleted: {file_path}")


# Load configuration from JSON file
config_file_path = 'config.json'

if not os.path.isfile(config_file_path):
    raise FileNotFoundError(f"Configuration file '{config_file_path}' not found.")

with open(config_file_path, 'r') as config_file:
    config = json.load(config_file)

# Extract the images_directory from the config
images_directory = config.get('images_directory', '')

# Check if the images_directory is provided
if not images_directory:
    raise ValueError("Images directory path is not specified in the configuration file.")

delete_png_files_in_directory(images_directory)
