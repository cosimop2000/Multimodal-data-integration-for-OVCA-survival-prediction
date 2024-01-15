import os

def delete_png_files_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg"):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted: {file_path}")

# Replace 'your_images_directory' with the actual path to your "images" directory
images_directory = 'C:\\Users\\pavon\\Documents\\Bioinformatics_project\\PLIP\\images'
delete_png_files_in_directory(images_directory)
