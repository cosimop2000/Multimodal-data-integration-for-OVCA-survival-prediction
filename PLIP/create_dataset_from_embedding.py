import os
import pandas as pd
import json
import configparser
import torch
parser = configparser.ConfigParser()
parser.read('Data/config.ini')


embeddings_dir = parser['embeddings']['embeddings_dir']
#Iterate over the folders in the embeddings directory
for subdir in os.listdir(embeddings_dir):
    subdir_path = os.path.join(embeddings_dir, subdir)
    if os.path.isdir(subdir_path):
        embedding_matrix = torch.tensor([])
        #Iterate over the files in the folder
        for filename in os.listdir(subdir_path):
            if filename.endswith(('.pt')):
                #Load the embedding
                embedding_path = os.path.join(subdir_path, filename)
                embedding = torch.load(embedding_path).squeeze(1)
                #Concatenate the embedding to the matrix
                embedding_matrix = torch.cat((embedding_matrix, embedding), 0)
        csv_name = subdir + '.csv'
        csv_path = os.path.join(parser['embeddings']['embeddings_dataset'], csv_name)
        #Save the matrix as a csv file
        pd.DataFrame(embedding_matrix.detach().numpy()).to_csv(csv_path, header=None, index=None)

                