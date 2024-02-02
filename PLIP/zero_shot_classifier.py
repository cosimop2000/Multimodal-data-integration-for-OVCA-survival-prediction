from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
import torch
import json
import configparser
import os
import pathlib
import numpy as np
prob_dict = {}

SINGLE = '/Users/jacksalici/Downloads/photo_5897626654967185077_y.jpg'

class ZeroShotClassifier:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("vinid/plip")
        self.processor = CLIPProcessor.from_pretrained("vinid/plip")
        self.text_model = CLIPTextModelWithProjection.from_pretrained("vinid/plip")

    def get_text_embeddings(self, text_list):
        inputs = self.processor(text=text_list, return_tensors="pt", padding=True)
        outputs = self.text_model(**inputs)
        return outputs.text_embeds.detach()

    def zero_shot_classification(self, image_embeds, text_embeds):    
        
        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        self.logit_scale = torch.nn.Parameter(torch.tensor(2.6592))

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()
        
        probs = logits_per_image.softmax(dim=1)
        
        return probs
    
    def get_image_embeddings(self, image):
        inputs = self.processor(text=[''], images=image, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        return outputs.image_embeds.cpu().unsqueeze(0)
    
    def get_all(self, image, text_list):
        model2 = CLIPModel.from_pretrained("vinid/plip")
        processor2 = CLIPProcessor.from_pretrained("vinid/plip")
        inputs = processor2(text=text_list, images=image, return_tensors="pt", padding=True)
        outputs = model2(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        return logits_per_image.softmax(dim=1)  


if __name__ == "__main__":
    text_list=['an image of tumoral ovarian tissue', 'an image of healthy ovarian tissue']
    zsc = ZeroShotClassifier() #n image of ovarian cancer tissue
    text_embeddings = zsc.get_text_embeddings(text_list)
    parser = configparser.ConfigParser()
    parser.read('Data/config.ini')


    if not SINGLE: 

        embeddings_dir = parser['embeddings']['embeddings_dir']
        
        for filename in pathlib.Path(embeddings_dir).glob('**/*.pt'):
            wsi_name = filename.parent.name
            patch_name = filename.name[:-len("_embedding.pt")]

            image_embeddings = torch.load(filename).detach()
            probs = zsc.zero_shot_classification(image_embeddings, text_embeddings)
    else:
        from PIL import Image
        image = Image.open(SINGLE)
        image_embeddings = zsc.get_image_embeddings(image).detach()
        probs = zsc.zero_shot_classification(image_embeddings, text_embeddings)
        print(probs)
        
        probs = zsc.get_all(image, text_list)
        print(probs)

    
