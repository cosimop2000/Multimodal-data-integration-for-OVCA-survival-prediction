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
from sklearn.manifold import TSNE



class dataset(data.Dataset):
    def __init__(self, methylation_path = 'methylation_table.csv', gene_expr_path = 'fpkm_protein_coding.csv', labels_path = 'dead.csv', impute = False, embeddings = None, rand = True):
        if embeddings:
            self.gene_expr, self.methylation, self.labels, self.embeddings, self.ids = self.load_data(methylation_path, gene_expr_path, labels_path, impute, embeddings, rand)
            self.rand = rand
        else:
            self.gene_expr, self.methylation, self.labels = self.load_data(methylation_path, gene_expr_path, labels_path, impute, embeddings, rand=False)
            self.embeddings = None

    def load_data(self, methylation_path, gene_expr_path, labels_path, impute, embeddings, rand):
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
                intersection = intersection.intersection(embeddings.keys())
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
            #imp_m = SimpleImputer(missing_values=np.nan, strategy='mean')
            imp_g = SimpleImputer(missing_values=np.nan, strategy='mean')
            #X_m = imp_m.fit_transform(X_m)
            X_g = imp_g.fit_transform(X_g)
        else:
            #remove the columns with nan values
            X_m = X_m[:,~np.isnan(X_m).any(axis=0)]
            X_g = X_g[:,~np.isnan(X_g).any(axis=0)]
            #remove columns with 0 variance
            # X_m = X_m[:,X_m.std(axis=0) != 0]
            # X_g = X_g[:,X_g.std(axis=0) != 0]
        #scale the values
        X_g = np.log2(X_g + np.finfo(np.float32).eps)
        scaler_m= StandardScaler()
        scaler_g= StandardScaler()
        X_m = scaler_m.fit_transform(X_m)
        X_g = scaler_g.fit_transform(X_g)
        #apply pca to the values
        #shuffle x and y accordingly
        perm = torch.randperm(X_m.shape[0])
        X_m = torch.from_numpy(X_m[perm])
        X_g = torch.from_numpy(X_g[perm])
        Y = torch.from_numpy(Y[perm]).long()
        if not embeddings:
            return X_g, X_m, Y
        else:
            for k in embeddings.keys():
                list = []
                maxes = []
                if not rand:
                    for file in embeddings[k]:
                        df = pd.read_csv(file)
                        ar = df.to_numpy()
                        top10 = np.zeros([10, df.shape[1]])
                        for i in range(10):
                            max = np.max(np.sum(ar, axis=1))
                            maxes.append(max)
                            amax = np.argmax(np.sum(ar, axis=1))
                            top10[i] = ar[amax]
                            ar = np.delete(ar, amax, axis=0)
                        list.append(top10)
                    embeddings[k] = list
                else:
                    list = []
                    for file in embeddings[k]:
                        df = pd.read_csv(file)
                        #scale the values using standard scaler
                        #scaler = StandardScaler()
                        #df = scaler.fit_transform(df)
                        #get the df back to a dataframe
                        df = df.to_numpy()
                        #apply softmax to the values along the axis 1
                        #df = np.exp(df) / np.sum(np.exp(df), axis=1).reshape([-1,1])
                        df = pd.DataFrame(df)
                        list.append(df)
                    embeddings[k] = list

            return X_g, X_m, Y, embeddings, ids

    def __getitem__(self, index):
        if self.embeddings:
            if not self.rand:
                list = self.embeddings[self.ids.iloc[index]]
                df = list[-1].reshape(-1)
                return self.gene_expr[index], self.methylation[index], df, self.labels[index]
            else:
                list = self.embeddings[self.ids.iloc[index]]
                df = random.choice(list)
                df = df.sample(n=10)
                df = df.to_numpy().reshape(-1)
                return self.gene_expr[index], self.methylation[index], df, self.labels[index]
        else:
            return self.gene_expr[index], self.methylation[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


class dataset_CNV(data.Dataset):
    def __init__(self, methylation_path = 'methylation_table.csv', gene_expr_path = 'expression_protein_coding_uq.csv', labels_path = 'dead.csv', cnv_path = 'cnv_nan_removed.csv'):
        self.gene_expr, self.methylation, self.cnv, self.labels = self.load_data(methylation_path, gene_expr_path, cnv_path, labels_path)

    def load_data(self, methylation_path, gene_expr_path, cnv_path, labels_path):
        df_methylation = pd.read_csv(methylation_path)
        df_gene_expr = pd.read_csv(gene_expr_path)
        df_cnv = pd.read_csv(cnv_path)
        labels = pd.read_csv(labels_path)
        #iterate through the labels DF rows
        intersection_m = df_methylation.columns.intersection(labels['id'])
        intersection_g = df_gene_expr.columns.intersection(labels['id'])
        intersection_c = df_cnv.columns.intersection(labels['id'])
        intersection = intersection_m.intersection(intersection_g)
        intersection = intersection.intersection(intersection_c)
        #keep just the rows in labels where the value of 'id' is in the intersection, transpose and reindex
        labels = labels[labels['id'].isin(intersection)]
        df_methylation=df_methylation[intersection].transpose().reindex(labels['id'])
        df_gene_expr=df_gene_expr[intersection].transpose().reindex(labels['id'])
        df_cnv=df_cnv[intersection].transpose().reindex(labels['id'])
        ids = labels['id']
        #drop the id column
        labels = labels.drop('id', axis=1)
        labels = labels.values
        Y = labels.astype(np.float32)
        X_m = df_methylation.values.astype(np.float32)
        X_g = df_gene_expr.values.astype(np.float32)
        X_c = df_cnv.values.astype(np.float32)
        #discretize the labels in order to have 2 balanced classes
        median = np.median(Y)
        Y =np.where(Y < 4*365, 0, 1)
        X_m = X_m[:,~np.isnan(X_m).any(axis=0)]
        X_g = X_g[:,~np.isnan(X_g).any(axis=0)]
        X_c = X_c[:,~np.isnan(X_c).any(axis=0)]
            #remove columns with 0 variance
            # X_m = X_m[:,X_m.std(axis=0) != 0]
            # X_g = X_g[:,X_g.std(axis=0) != 0]
        #scale the values
        scaler_m= StandardScaler()
        scaler_g= StandardScaler()
        scaler_c= StandardScaler()
        X_m = scaler_m.fit_transform(X_m)
        X_g = scaler_g.fit_transform(X_g)
        X_c = scaler_c.fit_transform(X_c)
        #shuffle x and y accordingly
        perm = torch.randperm(X_m.shape[0])
        X_m = torch.from_numpy(X_m[perm])
        X_g = torch.from_numpy(X_g[perm])
        X_c = torch.from_numpy(X_c[perm])
        Y = torch.from_numpy(Y[perm]).long()
        return X_g, X_m, X_c, Y
    def __getitem__(self, index):
        return self.gene_expr[index], self.methylation[index], self.cnv[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


class dataset_CNV_imgs(data.Dataset):
    def __init__(self, methylation_path = 'methylation_table.csv', gene_expr_path = 'expression_protein_coding_uq.csv', labels_path = 'dead.csv', cnv_path = 'cnv_nan_removed.csv', embeddings = None, rand = True):
        if embeddings:
            self.gene_expr, self.methylation, self.cnv, self.labels, self.embeddings, self.ids = self.load_data(methylation_path, gene_expr_path, labels_path, cnv_path, embeddings, rand)
            self.rand = rand
        else:
            self.gene_expr, self.methylation, self.labels = self.load_data(methylation_path, gene_expr_path, labels_path, cnv_path, embeddings, rand=False)
            self.embeddings = None

    def load_data(self, methylation_path, gene_expr_path, labels_path, cnv_path, embeddings, rand):
        df_methylation = pd.read_csv(methylation_path)
        df_gene_expr = pd.read_csv(gene_expr_path)
        df_cnv = pd.read_csv(cnv_path)
        labels = pd.read_csv(labels_path)
        #iterate through the labels DF rows
        intersection_m = df_methylation.columns.intersection(labels['id'])
        intersection_g = df_gene_expr.columns.intersection(labels['id'])
        intersection_c = df_cnv.columns.intersection(labels['id'])
        intersection = intersection_m.intersection(intersection_g)
        intersection = intersection.intersection(intersection_c)
        if embeddings:
            with open(embeddings) as json_file:
                embeddings = json.load(json_file)
            #keep only the ids in the intersection
            embeddings = {k: embeddings[k] for k in intersection}
        #keep just the rows in labels where the value of 'id' is in the intersection, transpose and reindex
        labels = labels[labels['id'].isin(intersection)]
        df_methylation=df_methylation[intersection].transpose().reindex(labels['id'])
        df_gene_expr=df_gene_expr[intersection].transpose().reindex(labels['id'])
        df_cnv=df_cnv[intersection].transpose().reindex(labels['id'])
        ids = labels['id']
        #drop the id column
        labels = labels.drop('id', axis=1)
        labels = labels.values
        Y = labels.astype(np.float32)
        X_m = df_methylation.values.astype(np.float32)
        X_g = df_gene_expr.values.astype(np.float32)
        X_c = df_cnv.values.astype(np.float32)
        #discretize the labels in order to have 2 balanced classes
        median = np.median(Y)
        Y =np.where(Y < 4*365, 0, 1)
        #remove the columns with nan values
        X_m = X_m[:,~np.isnan(X_m).any(axis=0)]
        X_g = X_g[:,~np.isnan(X_g).any(axis=0)]
        X_c = X_c[:,~np.isnan(X_c).any(axis=0)]
        #remove columns with 0 variance
        # X_m = X_m[:,X_m.std(axis=0) != 0]
        # X_g = X_g[:,X_g.std(axis=0) != 0]
        #scale the values
        scaler_m= StandardScaler()
        scaler_g= StandardScaler()
        scaler_c= StandardScaler()
        X_m = scaler_m.fit_transform(X_m)
        X_g = scaler_g.fit_transform(X_g)
        X_c = scaler_c.fit_transform(X_c)
        #shuffle x and y accordingly
        perm = torch.randperm(X_m.shape[0])
        X_m = torch.from_numpy(X_m[perm])
        X_g = torch.from_numpy(X_g[perm])
        X_c = torch.from_numpy(X_c[perm])
        Y = torch.from_numpy(Y[perm]).long()
        if not embeddings:
            return X_g, X_m, Y
        else:
            for k in embeddings.keys():
                list = []
                maxes = []
                if not rand:
                    for file in embeddings[k]:
                        df = pd.read_csv(file)
                        ar = df.to_numpy()
                        top10 = np.zeros([10, df.shape[1]])
                        for i in range(10):
                            max = np.max(np.sum(ar, axis=1))
                            maxes.append(max)
                            amax = np.argmax(np.sum(ar, axis=1))
                            top10[i] = ar[amax]
                            ar = np.delete(ar, amax, axis=0)
                        list.append(top10)
                    embeddings[k] = list
                else:
                    list = []
                    for file in embeddings[k]:
                        df = pd.read_csv(file)
                        #scale the values using standard scaler
                        #scaler = StandardScaler()
                        #df = scaler.fit_transform(df)
                        #get the df back to a dataframe
                        df = pd.DataFrame(df)
                        list.append(df)
                    embeddings[k] = list
            return X_g, X_m, X_c, Y, embeddings, ids

    def __getitem__(self, index):
        if self.embeddings:
            if not self.rand:
                list = self.embeddings[self.ids.iloc[index]]
                df = list[-1].reshape(-1)
                return self.gene_expr[index], self.methylation[index], self.cnv[index], df, self.labels[index]
            else:
                list = self.embeddings[self.ids.iloc[index]]
                df = random.choice(list)
                df = df.sample(n=10)
                df = df.to_numpy().reshape(-1)
                return self.gene_expr[index], self.methylation[index], self.cnv[index], df, self.labels[index]
        else:
            return self.gene_expr[index], self.methylation[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


class dataset_met_img(data.Dataset):
    def __init__(self, methylation_path = 'methylation_table.csv', gene_expr_path = 'fpkm_protein_coding.csv', labels_path = 'dead.csv', impute = False, embeddings = None, rand = True):
        if embeddings:
            self.methylation, self.labels, self.embeddings, self.ids = self.load_data(methylation_path, gene_expr_path, labels_path, impute, embeddings, rand)
            self.rand = rand
        else:
            self.methylation, self.labels = self.load_data(methylation_path, gene_expr_path, labels_path, impute, embeddings, rand=False)
            self.embeddings = None

    def load_data(self, methylation_path, gene_expr_path, labels_path, impute, embeddings, rand):
        df_methylation = pd.read_csv(methylation_path)
        df_gene_expr = pd.read_csv(gene_expr_path)
        labels = pd.read_csv(labels_path)
        #iterate through the labels DF rows
        intersection = df_methylation.columns.intersection(labels['id'])
        # intersection_g = df_gene_expr.columns.intersection(labels['id'])
        # intersection = intersection_m.intersection(intersection_g)
        if embeddings:
            with open(embeddings) as json_file:
                embeddings = json.load(json_file)
            #keep only the ids in the intersection
                intersection = intersection.intersection(embeddings.keys())
            embeddings = {k: embeddings[k] for k in intersection}
        #keep just the rows in labels where the value of 'id' is in the intersection, transpose and reindex
        labels = labels[labels['id'].isin(intersection)]
        df_methylation=df_methylation[intersection].transpose().reindex(labels['id'])
        #df_gene_expr=df_gene_expr[intersection].transpose().reindex(labels['id'])
        ids = labels['id']
        #drop the id column
        labels = labels.drop('id', axis=1)
        labels = labels.values
        Y = labels.astype(np.float32)
        X_m = df_methylation.values.astype(np.float32)
        #X_g = df_gene_expr.values.astype(np.float32)
        #discretize the labels in order to have 2 balanced classes
        median = np.median(Y)
        Y =np.where(Y < 4*365, 0, 1)
        if impute:
            imp_m = SimpleImputer(missing_values=np.nan, strategy='mean')
            #imp_g = SimpleImputer(missing_values=np.nan, strategy='mean')
            X_m = imp_m.fit_transform(X_m)
            #X_g = imp_g.fit_transform(X_g)
        else:
            #remove the columns with nan values
            X_m = X_m[:,~np.isnan(X_m).any(axis=0)]
            #X_g = X_g[:,~np.isnan(X_g).any(axis=0)]
            #remove columns with 0 variance
            # X_m = X_m[:,X_m.std(axis=0) != 0]
            # X_g = X_g[:,X_g.std(axis=0) != 0]
        #scale the values
        scaler_m= StandardScaler()
        #scaler_g= StandardScaler()
        X_m = scaler_m.fit_transform(X_m)
        #X_g = scaler_g.fit_transform(X_g)
        #shuffle x and y accordingly
        perm = torch.randperm(X_m.shape[0])
        X_m = torch.from_numpy(X_m[perm]).float()
        #X_g = torch.from_numpy(X_g[perm])
        Y = torch.from_numpy(Y[perm]).long()
        if not embeddings:
            return X_m, Y
        else:
            for k in embeddings.keys():
                list = []
                maxes = []
                if not rand:
                    for file in embeddings[k]:
                        df = pd.read_csv(file)
                        ar = df.to_numpy()
                        top10 = np.zeros([10, df.shape[1]])
                        for i in range(10):
                            max = np.max(np.sum(ar, axis=1))
                            maxes.append(max)
                            amax = np.argmax(np.sum(ar, axis=1))
                            top10[i] = ar[amax]
                            ar = np.delete(ar, amax, axis=0)
                        list.append(top10)
                    embeddings[k] = list
                else:
                    list = []
                    for file in embeddings[k]:
                        df = pd.read_csv(file)
                        #scale the values using standard scaler
                        #scaler = StandardScaler()
                        #df = scaler.fit_transform(df)
                        #get the df back to a dataframe
                        df = df.to_numpy()
                        #apply softmax to the values along the axis 1
                        #df = np.exp(df) / np.sum(np.exp(df), axis=1).reshape([-1,1])
                        df = pd.DataFrame(df)
                        list.append(df)
                    embeddings[k] = list
            return X_m, Y, embeddings, ids

    def __getitem__(self, index):
        if self.embeddings:
            if not self.rand:
                list = self.embeddings[self.ids.iloc[index]]
                df = list[-1].reshape(-1)
                return self.methylation[index], df, self.labels[index]
            else:
                list = self.embeddings[self.ids.iloc[index]]
                df = random.choice(list)
                df = df.sample(n=10)
                df = torch.from_numpy(df.to_numpy().reshape(-1)).float()
                return self.methylation[index], df, self.labels[index]
        else:
            return self.methylation[index], self.labels[index]

    def __len__(self):
        return len(self.labels)
    

class dataset_gene_expr_img(data.Dataset):
    def __init__(self, methylation_path = 'methylation_table.csv', gene_expr_path = 'fpkm_protein_coding.csv', labels_path = 'dead.csv', impute = False, embeddings = None, rand = True):
        if embeddings:
            self.gene_expr, self.labels, self.embeddings, self.ids = self.load_data(methylation_path, gene_expr_path, labels_path, impute, embeddings, rand)
            self.rand = rand
        else:
            self.gene_expr, self.methylation, self.labels = self.load_data(methylation_path, gene_expr_path, labels_path, impute, embeddings, rand=False)
            self.embeddings = None

    def load_data(self, methylation_path, gene_expr_path, labels_path, impute, embeddings, rand):
        #df_methylation = pd.read_csv(methylation_path)
        df_gene_expr = pd.read_csv(gene_expr_path)
        labels = pd.read_csv(labels_path)
        #iterate through the labels DF rows
        #intersection_m = df_methylation.columns.intersection(labels['id'])
        intersection = df_gene_expr.columns.intersection(labels['id'])
        #intersection = intersection_m.intersection(intersection_g)
        if embeddings:
            with open(embeddings) as json_file:
                embeddings = json.load(json_file)
            #keep only the ids in the intersection
                intersection = intersection.intersection(embeddings.keys())
            embeddings = {k: embeddings[k] for k in intersection}
        #keep just the rows in labels where the value of 'id' is in the intersection, transpose and reindex
        labels = labels[labels['id'].isin(intersection)]
        #df_methylation=df_methylation[intersection].transpose().reindex(labels['id'])
        df_gene_expr=df_gene_expr[intersection].transpose().reindex(labels['id'])
        ids = labels['id']
        #drop the id column
        labels = labels.drop('id', axis=1)
        labels = labels.values
        Y = labels.astype(np.float32)
        #X_m = df_methylation.values.astype(np.float32)
        X_g = df_gene_expr.values.astype(np.float32)
        #discretize the labels in order to have 2 balanced classes
        median = np.median(Y)
        Y =np.where(Y < 4*365, 0, 1)
        if impute:
            #imp_m = SimpleImputer(missing_values=np.nan, strategy='mean')
            imp_g = SimpleImputer(missing_values=np.nan, strategy='mean')
            #X_m = imp_m.fit_transform(X_m)
            X_g = imp_g.fit_transform(X_g)
        else:
            #remove the columns with nan values
            #X_m = X_m[:,~np.isnan(X_m).any(axis=0)]
            X_g = X_g[:,~np.isnan(X_g).any(axis=0)]
            #remove columns with 0 variance
            # X_m = X_m[:,X_m.std(axis=0) != 0]
            # X_g = X_g[:,X_g.std(axis=0) != 0]
        #scale the values
        #scaler_m= StandardScaler()
        scaler_g= StandardScaler()
        #X_m = scaler_m.fit_transform(X_m)
        X_g = scaler_g.fit_transform(X_g)
        #shuffle x and y accordingly
        perm = torch.randperm(X_g.shape[0])
        #X_m = torch.from_numpy(X_m[perm])
        X_g = torch.from_numpy(X_g[perm])
        Y = torch.from_numpy(Y[perm]).long()
        if not embeddings:
            return X_g, Y
        else:
            for k in embeddings.keys():
                list = []
                maxes = []
                if not rand:
                    for file in embeddings[k]:
                        df = pd.read_csv(file)
                        ar = df.to_numpy()
                        top10 = np.zeros([10, df.shape[1]])
                        for i in range(10):
                            max = np.max(np.sum(ar, axis=1))
                            maxes.append(max)
                            amax = np.argmax(np.sum(ar, axis=1))
                            top10[i] = ar[amax]
                            ar = np.delete(ar, amax, axis=0)
                        list.append(top10)
                    embeddings[k] = list
                else:
                    list = []
                    for file in embeddings[k]:
                        df = pd.read_csv(file)
                        #scale the values using standard scaler
                        #scaler = StandardScaler()
                        #df = scaler.fit_transform(df)
                        #get the df back to a dataframe
                        df = df.to_numpy()
                        #apply softmax to the values along the axis 1
                        #df = np.exp(df) / np.sum(np.exp(df), axis=1).reshape([-1,1])
                        df = pd.DataFrame(df)
                        list.append(df)
                    embeddings[k] = list
            return X_g, Y, embeddings, ids

    def __getitem__(self, index):
        if self.embeddings:
            if not self.rand:
                list = self.embeddings[self.ids.iloc[index]]
                df = list[-1].reshape(-1)
                return self.gene_expr[index], self.methylation[index], df, self.labels[index]
            else:
                list = self.embeddings[self.ids.iloc[index]]
                df = random.choice(list)
                df = df.sample(n=10)
                df = torch.from_numpy(df.to_numpy().reshape(-1)).float()
                return self.gene_expr[index], df, self.labels[index]
        else:
            return self.gene_expr[index], self.labels[index]

    def __len__(self):
        return len(self.labels)
    

class dataset_miRNA(data.Dataset):
    def __init__(self, methylation_path = 'methylation_table.csv', gene_expr_path = 'expression_protein_coding_uq.csv', labels_path = 'dead.csv', cnv_path = 'miRNA.csv'):
        self.gene_expr, self.methylation, self.cnv, self.labels = self.load_data(methylation_path, gene_expr_path, cnv_path, labels_path)

    def load_data(self, methylation_path, gene_expr_path, cnv_path, labels_path):
        df_methylation = pd.read_csv(methylation_path)
        df_gene_expr = pd.read_csv(gene_expr_path)
        df_cnv = pd.read_csv(cnv_path)
        labels = pd.read_csv(labels_path)
        #iterate through the labels DF rows
        intersection_m = df_methylation.columns.intersection(labels['id'])
        intersection_g = df_gene_expr.columns.intersection(labels['id'])
        intersection_c = df_cnv.columns.intersection(labels['id'])
        intersection = intersection_m.intersection(intersection_g)
        intersection = intersection.intersection(intersection_c)
        #keep just the rows in labels where the value of 'id' is in the intersection, transpose and reindex
        labels = labels[labels['id'].isin(intersection)]
        df_methylation=df_methylation[intersection].transpose().reindex(labels['id'])
        df_gene_expr=df_gene_expr[intersection].transpose().reindex(labels['id'])
        df_cnv=df_cnv[intersection].transpose().reindex(labels['id'])
        ids = labels['id']
        #drop the id column
        labels = labels.drop('id', axis=1)
        labels = labels.values
        Y = labels.astype(np.float32)
        X_m = df_methylation.values.astype(np.float32)
        X_g = df_gene_expr.values.astype(np.float32)
        X_c = df_cnv.values.astype(np.float32)
        #discretize the labels in order to have 2 balanced classes
        median = np.median(Y)
        Y =np.where(Y < 4*365, 0, 1)
        X_m = X_m[:,~np.isnan(X_m).any(axis=0)]
        X_g = X_g[:,~np.isnan(X_g).any(axis=0)]
        X_c = X_c[:,~np.isnan(X_c).any(axis=0)]
            #remove columns with 0 variance
            # X_m = X_m[:,X_m.std(axis=0) != 0]
            # X_g = X_g[:,X_g.std(axis=0) != 0]
        #scale the values
        scaler_m= StandardScaler()
        scaler_g= StandardScaler()
        scaler_c= StandardScaler()
        X_m = scaler_m.fit_transform(X_m)
        X_g = scaler_g.fit_transform(X_g)
        X_c = scaler_c.fit_transform(X_c)
        #shuffle x and y accordingly
        perm = torch.randperm(X_m.shape[0])
        X_m = torch.from_numpy(X_m[perm])
        X_g = torch.from_numpy(X_g[perm])
        X_c = torch.from_numpy(X_c[perm])
        Y = torch.from_numpy(Y[perm]).long()
        return X_g, X_m, X_c, Y
    def __getitem__(self, index):
        return self.gene_expr[index], self.methylation[index], self.cnv[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

class dataset_gene_meth_better(data.Dataset):
    def __init__(self, methylation_path = 'TCGA-OV.methylation27.tsv', gene_expr_path = 'TCGA-OV.htseq_fpkm-uq.tsv', labels_path = 'TCGA-OV.survival_labeled.tsv'):
        self.gene_expr, self.methylation, self.labels = self.load_data(methylation_path, gene_expr_path, labels_path)

    def load_data(self, methylation_path, gene_expr_path, labels_path):
        df_methylation = pd.read_csv(methylation_path, sep='\t')
        df_gene_expr = pd.read_csv(gene_expr_path, sep='\t')
        labels = pd.read_csv(labels_path, sep='\t')
        #iterate through the labels DF rows
        labels=labels.loc[labels['OS']==1]
        intersection_m = df_methylation.columns.intersection(labels['sample'])
        intersection_g = df_gene_expr.columns.intersection(labels['sample'])
        intersection = intersection_m.intersection(intersection_g)
        #keep just the rows in labels where the value of 'id' is in the intersection, transpose and reindex
        labels = labels[labels['sample'].isin(intersection)]
        df_methylation=df_methylation[intersection].transpose().reindex(labels['sample'])
        df_gene_expr=df_gene_expr[intersection].transpose().reindex(labels['sample'])
        ids = labels['sample']
        #drop the id column
        labels = labels['OS.label']
        labels = labels.values
        Y = labels.astype(np.float32)
        X_m = df_methylation.values.astype(np.float32)
        X_g = df_gene_expr.values.astype(np.float32)
        #discretize the labels in order to have 2 balanced classes
        median = np.median(Y)
        #Y =np.where(Y < 4*365, 0, 1)
        X_m = X_m[:,~np.isnan(X_m).any(axis=0)]
        X_g = X_g[:,~np.isnan(X_g).any(axis=0)]
            #remove columns with 0 variance
            # X_m = X_m[:,X_m.std(axis=0) != 0]
            # X_g = X_g[:,X_g.std(axis=0) != 0]
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
        return X_g, X_m, Y
    def __getitem__(self, index):
        return self.gene_expr[index], self.methylation[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    dataset = dataset_gene_meth_better()
    sample = dataset[0]
    print(len(dataset[0]))
