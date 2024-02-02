from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPTextModel
import torch
import json
import configparser
import os
import pathlib
import numpy as np
prob_dict = {}

class ZeroShotClassifier:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("vinid/plip")
        self.processor = CLIPProcessor.from_pretrained("vinid/plip")
        self.text_model = CLIPTextModel.from_pretrained("vinid/plip")

    def get_text_embeddings(self, text_list):
        inputs = self.processor(text=text_list, return_tensors="pt", padding=True)
        outputs = self.text_model(**inputs)
        return outputs.pooler_output.detach().numpy()

    def zero_shot_classification(self, image_embeddings, text_embeddings):    
        image_embeddings = image_embeddings/np.linalg.norm(image_embeddings, ord=2, axis=-1, keepdims=True)
        text_embeddings = text_embeddings/np.linalg.norm(text_embeddings, ord=2, axis=-1, keepdims=True)
        
        sim = image_embeddings.dot(text_embeddings.T).squeeze()
        sim = sim/np.linalg.norm(sim, ord=2, axis=-1, keepdims=True)

        
        probs = torch.softmax(torch.from_numpy(sim), 0)  # we can take the softmax to get the label probabilities
        
        return probs


if __name__ == "__main__":
    zsc = ZeroShotClassifier() #n image of ovarian cancer tissue
    text_embeddings = zsc.get_text_embeddings(text_list=['an image of healthy ovarian tissue', 'a dog'])
    parser = configparser.ConfigParser()
    parser.read('Data/config.ini')


    embeddings_dir = parser['embeddings']['embeddings_dir']
    
    for filename in pathlib.Path(embeddings_dir).glob('**/*.pt'):
        wsi_name = filename.parent.name
        patch_name = filename.name[:-len("_embedding.pt")]

        image_embeddings = torch.load(filename).detach().numpy()
        

        image_probs = zsc.zero_shot_classification(image_embeddings, text_embeddings)
        print(image_probs)
            

    
