from PIL import Image
import os
import torch
from transformers import CLIPProcessor, CLIPModel


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

    input_root_dir = "C:\\Users\\pavon\\Documents\\Bioinformatics_project\\PLIP\\images"
    output_root_dir = "C:\\Users\\pavon\\Documents\\Bioinformatics_project\\PLIP\\embeddings"

    for subdir in os.listdir(input_root_dir):
        subdir_path = os.path.join(input_root_dir, subdir)
        if os.path.isdir(subdir_path):
            output_dir = os.path.join(output_root_dir, subdir)

            # Create output directory if not exists
            os.makedirs(output_dir, exist_ok=True)

            process_images_and_save_embeddings(model.to(device), processor, subdir_path, output_dir, device)
