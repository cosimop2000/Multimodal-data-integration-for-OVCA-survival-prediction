import pandas as pd
import numpy as np
import torch
import sklearn
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import torch_geometric


def load_dataset(df, labels, task='classification', pca_components=300):
    #iterate through the labels DF rows
    intersection = df.columns.intersection(labels['id'])
    df = df[intersection]
    #keep just the rows in labels where the value of 'id' is in the intersection
    labels = labels[labels['id'].isin(intersection)]
    df=df[intersection]
    df = df.transpose()
    #order the df by the labels
    df = df.reindex(labels['id'])
    #drop the id column
    labels = labels.drop('id', axis=1)
    labels = labels.values
    Y = labels.astype(np.float32)
    X = df.values.astype(np.float32)
    if task == 'classification':
        #scale Y between 0 and 1
        Y = (Y-np.min(Y))/(np.max(Y)-np.min(Y))
        #discretize Y in 3 classes:
        Y = np.where(Y<0.2, 0, Y)
        Y = np.where(Y>0.5, 2, Y)
        Y = np.where((Y>=0.2) & (Y<=0.5), 1, Y)
    #use sklearn pca to reduce the dimensionality of x to 500
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    X = imp.fit_transform(X)
    if pca_components > 0:
        pca = PCA(n_components=pca_components)
        X = pca.fit_transform(X)
    #Scale between 0 and 1
    X = (X-np.min(X))/(np.max(X)-np.min(X))
    #normalize X to 0 mean and 1 variance
    X = sklearn.preprocessing.scale(X)
    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)
    if task == 'classification':
        Y = Y.long()
    return X, Y


def get_graph(X, y, train_size=0.8, correlation_threshold=0.0):
    correlation = torch.from_numpy(np.where(torch.tensor(np.corrcoef(X))>correlation_threshold, 1, 0))
    for i in range(correlation.shape[0]):
        correlation[i, i] = 0
    #make correlation a sparse matrix
    edge_index = torch.nonzero(correlation).t()
    data = torch_geometric.data.Data(x=X, edge_index=edge_index, y=y)
    #create train and test masks
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[:int(train_size*data.num_nodes)] = 1
    test_mask[int(train_size*data.num_nodes):] = 1
    data.train_mask = train_mask
    data.test_mask = test_mask
    return data

def test_GNN(model, data, task='classification', verbose=False):
    model.eval()
    out = model(data.x, data.edge_index)
    test_acc = 0
    if task == 'classification':
        out = torch.softmax(out, dim=1)
        out = torch.argmax(out, dim=1) # Use the class with highest probability.
        test_correct = out[data.test_mask] == data.y[data.test_mask].reshape(-1) # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
    if task == 'regression':
        #compute MSE
        loss = torch.nn.L1Loss()
        test_acc = loss(out[data.test_mask],data.y[data.test_mask].reshape(-1)) 
    if verbose:
        print('predicitions: ', out[data.test_mask])
        print('ground truth: ', data.y[data.test_mask].reshape(-1))
    return test_acc

