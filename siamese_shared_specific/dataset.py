import torch.utils.data as data
import torch
import pandas as pd
import sklearn
import numpy as np

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer



class dataset(data.Dataset):
    def __init__(self, methylation_path = 'methylation_table.csv', gene_expr_path = 'fpkm.csv', labels_path = 'complete.csv'):
        self.gene_expr, self.methylation, self.labels = self.load_data(methylation_path, gene_expr_path, labels_path)

    def load_data(self, methylation_path, gene_expr_path, labels_path):
        df_methylation = pd.read_csv(methylation_path)
        df_gene_expr = pd.read_csv(gene_expr_path)
        labels = pd.read_csv(labels_path)
        #iterate through the labels DF rows
        intersection_m = df_methylation.columns.intersection(labels['id'])
        intersection_g = df_gene_expr.columns.intersection(labels['id'])
        intersection = intersection_m.intersection(intersection_g)
        #keep just the rows in labels where the value of 'id' is in the intersection, transpose and reindex
        labels = labels[labels['id'].isin(intersection)]
        df_methylation=df_methylation[intersection].transpose().reindex(labels['id'])
        df_gene_expr=df_gene_expr[intersection].transpose().reindex(labels['id'])
        #drop the id column
        labels = labels.drop('id', axis=1)
        labels = labels.values
        Y = labels.astype(np.float32)
        X_m = df_methylation.values.astype(np.float32)
        X_g = df_gene_expr.values.astype(np.float32)
        Y =np.where(Y < 4*365, 0, 1)
        imp_m = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_g = SimpleImputer(missing_values=np.nan, strategy='mean')
        X_m = imp_m.fit_transform(X_m)
        X_g = imp_g.fit_transform(X_g)
        #shuffle x and y accordingly
        perm = torch.randperm(X_m.shape[0])
        X_m = torch.from_numpy(X_m[perm])
        X_g = torch.from_numpy(X_g[perm])
        Y = torch.from_numpy(Y[perm]).long()
        return X_g, X_m, Y

    def __getitem__(self, index):
        return self.gene_expr[index], self.methylation[index], self.labels[index]

    def __len__(self):
        return len(self.labels)
    
if __name__ == '__main__':
    dataset = dataset()
    print(len(dataset[0]))