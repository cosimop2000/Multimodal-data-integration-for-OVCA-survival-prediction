import os
import shutil

source_directory = "C:\\Users\\pavon\\Desktop\\WSI"
destination_directory = "C:\\Users\\pavon\\Documents\\Bioinformatics_project\\Data\\WSI\\DATA_DIRECTORY"

for root, dirs, files in os.walk(source_directory):
    if "log" in dirs:
        dirs.remove("logs")  # Exclude the "log" directory from further processing

    for file in files:
        if file.endswith(".svs"):
            source_path = os.path.join(root, file)
            destination_path = os.path.join(destination_directory, file)
            shutil.move(source_path, destination_path)

print("Files moved successfully.")
