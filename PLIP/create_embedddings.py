from PIL import Image
import os
import torch
from transformers import CLIPProcessor, CLIPModel
import json


def process_images_and_save_embeddings(model, processor, image_dir, output_dir, device):
    for filename in os.listdir(image_dir):
        if filename.endswith(('.jpg', '.png')):
            image_path = os.path.join(image_dir, filename)
            image = Image.open(image_path)

            inputs = processor(text=[''], images=image, return_tensors="pt", padding=True)

            # Move inputs to GPU
            inputs = {key: value.to(device) for key, value in inputs.items()}

            outputs = model(**inputs)

            # Move embeddings back to CPU before saving
            embedding_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_embedding.pt")
            torch.save(outputs.image_embeds.cpu().unsqueeze(0), embedding_path)


if __name__ == "__main__":
    model = CLIPModel.from_pretrained("vinid/plip")
    processor = CLIPProcessor.from_pretrained("vinid/plip")

    # Specify the device (cuda or cpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load configuration from JSON file
    config_file_path = 'config.json'

    if not os.path.isfile(config_file_path):
        raise FileNotFoundError(f"Configuration file '{config_file_path}' not found.")

    with open(config_file_path, 'r') as config_file:
        config = json.load(config_file)

    # Extract the paths from the config
    input_root_dir = config.get('input_root_dir', '')
    output_root_dir = config.get('output_root_dir', '')

    # Check if the paths are provided
    if not input_root_dir or not output_root_dir:
        raise ValueError("Input or output root directory is not specified in the configuration file.")

    for subdir in os.listdir(input_root_dir):
        subdir_path = os.path.join(input_root_dir, subdir)
        if os.path.isdir(subdir_path):
            output_dir = os.path.join(output_root_dir, subdir)

            # Create output directory if not exists
            os.makedirs(output_dir, exist_ok=True)

            process_images_and_save_embeddings(model.to(device), processor, subdir_path, output_dir, device)
