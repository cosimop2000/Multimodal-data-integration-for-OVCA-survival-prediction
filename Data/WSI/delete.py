import os

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
    # Replace with the actual paths
    svs_root_directory = 'C:\\Users\\pavon\\Documents\\Bioinformatics_project\\Data\\WSI\\DATA_DIRECTORY'
    h5_root_directory = 'C:\\Users\\pavon\\Documents\\Bioinformatics_project\\Data\\WSI\\RESULTS_DIRECTORY\\patches'
    patches_directory1 = 'C:\\Users\\pavon\\Documents\\Bioinformatics_project\\Data\\WSI\\RESULTS_DIRECTORY\\masks'
    patches_directory2 = 'C:\\Users\\pavon\\Documents\\Bioinformatics_project\\Data\\WSI\\RESULTS_DIRECTORY\\stitches'
    # Uncomment the lines below to execute each function
    delete_svs_files(svs_root_directory)
    delete_h5_files(h5_root_directory)
    delete_patches_png(patches_directory1)
    delete_patches_png(patches_directory2)