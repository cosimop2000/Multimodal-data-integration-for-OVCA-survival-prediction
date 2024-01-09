from PIL import Image
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("vinid/plip")
processor = CLIPProcessor.from_pretrained("vinid/plip")

image = Image.open("images/image1.jpg")

inputs = processor(text=[''],
                   images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)

print(outputs.image_embeds.unsqueeze(0))
