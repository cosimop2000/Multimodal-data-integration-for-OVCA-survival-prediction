import pandas as pd
import numpy as np
import torch
import sklearn
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import torch_geometric
from torch.nn import functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.aggr import SoftmaxAggregation
from torch_geometric.utils import to_networkx
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import svm



if __name__ == '__main__':
    X = pd.read_csv('expression.tsv', sep='\t', header=0, index_col=0)
    y = pd.read_csv('labels.tsv', sep='\t', header=0)
    
    X = X[X.columns.intersection(y['id'])].transpose()
    print(y.describe())
    print("Original X shape:", X.shape)
    
    pca = PCA(n_components=250)
    X = pca.fit_transform(X, y['daysToDeath'])
    
    print("X shape after PCA:", X.shape)
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y['daysToDeath'], random_state=1)
    clfs = [MLPRegressor(random_state=1, max_iter=500, verbose=True, hidden_layer_sizes=[100], early_stopping=True),
            SVC()]

    
    
    
    clfs[0].fit(X_train, y_train)
    y_pred = clfs[0].predict(X_test)
    print(mean_absolute_error(y_pred, y_test))
    
        
    #plt.plot(clfs[2].loss_curve_)
    #plt.show()

    
    
    
