import os
import json

def delete_svs_files(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.svs'):
                file_path = os.path.join(dirpath, filename)
                os.remove(file_path)
                print(f"Deleted: {file_path}")

def delete_h5_files(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.h5'):
                file_path = os.path.join(dirpath, filename)
                os.remove(file_path)
                print(f"Deleted: {file_path}")

def delete_patches_png(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith('.jpg'):
            file_path = os.path.join(directory, filename)
            os.remove(file_path)
            print(f"Deleted: {file_path}")

if __name__ == "__main__":
    # Load configuration from JSON file
    config_file_path = 'config.json'

    if not os.path.isfile(config_file_path):
        raise FileNotFoundError(f"Configuration file '{config_file_path}' not found.")

    with open(config_file_path, 'r') as config_file:
        config = json.load(config_file)

    # Extract the paths from the configuration
    svs_root_directory = config.get('svs_root_directory', '')
    h5_root_directory = config.get('h5_root_directory', '')
    patches_directory1 = config.get('patches_directory1', '')
    patches_directory2 = config.get('patches_directory2', '')

    # Check if the paths are provided
    if not all([svs_root_directory, h5_root_directory, patches_directory1, patches_directory2]):
        raise ValueError("One or more paths are not specified in the configuration file.")
    delete_svs_files(svs_root_directory)
    delete_h5_files(h5_root_directory)
    delete_patches_png(patches_directory1)
    delete_patches_png(patches_directory2)