import os
import shutil
import json

# Load configuration from JSON file
config_file_path = 'config.json'

if not os.path.isfile(config_file_path):
    raise FileNotFoundError(f"Configuration file '{config_file_path}' not found.")

with open(config_file_path, 'r') as config_file:
    config = json.load(config_file)

# Extract values from the configuration

# Directory (path) where all the directories containing .svs files are
source_directory = config.get('source_directory', '')

# DATA_DIRECTORY
destination_directory = config.get('destination_directory', '')

# Check if source and destination directories are provided
if not source_directory or not destination_directory:
    raise ValueError("Source or destination directory is not specified in the configuration file.")
for root, dirs, files in os.walk(source_directory):
    if "logs" in dirs:
        dirs.remove("logs")  # Exclude the "log" directory from further processing

    for file in files:
        if file.endswith(".svs"):
            source_path = os.path.join(root, file)
            destination_path = os.path.join(destination_directory, file)
            shutil.move(source_path, destination_path)

print("Files moved successfully.")
