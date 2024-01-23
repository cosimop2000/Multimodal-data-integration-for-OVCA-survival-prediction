import torch.utils.data as data
import torch
import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import json
import random



class dataset(data.Dataset):
    def __init__(self, methylation_path = 'methylation_table.csv', gene_expr_path = 'fpkm_protein_coding.csv', labels_path = 'dead.csv', impute = False, embeddings = None):
        if embeddings:
            self.gene_expr, self.methylation, self.labels, self.embeddings, self.ids = self.load_data(methylation_path, gene_expr_path, labels_path, impute, embeddings)
        else:
            self.gene_expr, self.methylation, self.labels = self.load_data(methylation_path, gene_expr_path, labels_path, impute, embeddings)
            self.embeddings = None

    def load_data(self, methylation_path, gene_expr_path, labels_path, impute, embeddings):
        df_methylation = pd.read_csv(methylation_path)
        df_gene_expr = pd.read_csv(gene_expr_path)
        labels = pd.read_csv(labels_path)
        #iterate through the labels DF rows
        intersection_m = df_methylation.columns.intersection(labels['id'])
        intersection_g = df_gene_expr.columns.intersection(labels['id'])
        intersection = intersection_m.intersection(intersection_g)
        if embeddings:
            with open(embeddings) as json_file:
                embeddings = json.load(json_file)
            #keep only the ids in the intersection
            embeddings = {k: embeddings[k] for k in intersection}
        #keep just the rows in labels where the value of 'id' is in the intersection, transpose and reindex
        labels = labels[labels['id'].isin(intersection)]
        df_methylation=df_methylation[intersection].transpose().reindex(labels['id'])
        df_gene_expr=df_gene_expr[intersection].transpose().reindex(labels['id'])
        ids = labels['id']
        #drop the id column
        labels = labels.drop('id', axis=1)
        labels = labels.values
        Y = labels.astype(np.float32)
        X_m = df_methylation.values.astype(np.float32)
        X_g = df_gene_expr.values.astype(np.float32)
        #discretize the labels in order to have 2 balanced classes
        median = np.median(Y)
        Y =np.where(Y < 4*365, 0, 1)
        if impute:
            imp_m = SimpleImputer(missing_values=np.nan, strategy='mean')
            imp_g = SimpleImputer(missing_values=np.nan, strategy='mean')
            X_m = imp_m.fit_transform(X_m)
            X_g = imp_g.fit_transform(X_g)
        else:
            #remove the columns with nan values
            X_m = X_m[:,~np.isnan(X_m).any(axis=0)]
            X_g = X_g[:,~np.isnan(X_g).any(axis=0)]
            #remove columns with all values equal to 0
            # X_m = X_m[:,~np.all(X_m == 0, axis=0)]
            # X_g = X_g[:,~np.all(X_g == 0, axis=0)]
        #scale the values
        scaler_m= StandardScaler()
        scaler_g= StandardScaler()
        X_m = scaler_m.fit_transform(X_m)
        X_g = scaler_g.fit_transform(X_g)
        #shuffle x and y accordingly
        perm = torch.randperm(X_m.shape[0])
        X_m = torch.from_numpy(X_m[perm])
        X_g = torch.from_numpy(X_g[perm])
        Y = torch.from_numpy(Y[perm]).long()
        if not embeddings:
            return X_g, X_m, Y
        else:
            return X_g, X_m, Y, embeddings, ids

    def __getitem__(self, index):
        if self.embeddings:
            list = self.embeddings[self.ids.iloc[index]]
            embed = random.choice(list)
            df = pd.read_csv(embed)
            #select 10 random rows from df
            df = df.sample(n=10)
            df = df.to_numpy()
            df = df.reshape(-1)
            return self.gene_expr[index], self.methylation[index], df, self.labels[index]
        else:
            return self.gene_expr[index], self.methylation[index], self.labels[index]

    def __len__(self):
        return len(self.labels)




if __name__ == '__main__':
    dataset = dataset()
    sample = dataset[0]
    print(len(dataset[0]))