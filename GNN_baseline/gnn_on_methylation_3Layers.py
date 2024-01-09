import pandas as pd
import numpy as np
import torch
import sklearn
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import torch_geometric
from torch.nn import functional as F
from torch_geometric.nn import GCNConv

#Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    out = torch.softmax(out, dim=1)
    pred = torch.argmax(out, dim=1) # Use the class with highest probability.
    print('predicitions: ', pred[data.test_mask])
    print('ground truth: ', data.y[data.test_mask].reshape(-1))
    test_correct = pred[data.test_mask] == data.y[data.test_mask].reshape(-1) # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
    return test_acc

def load_dataset(df, labels):
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
    #scale Y between 0 and 1
    Y = (Y-np.min(Y))/(np.max(Y)-np.min(Y))
    #discretize Y in 3 classes:
    Y = np.where(Y<0.2, 0, Y)
    Y = np.where(Y>0.5, 2, Y)
    Y = np.where((Y>=0.2) & (Y<=0.5), 1, Y)
    #print the percatage of each class
    #use sklearn pca to reduce the dimensionality of x to 500
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    X = imp.fit_transform(X)
    pca = PCA(n_components=300)
    X = pca.fit_transform(X)
    #Scale between 0 and 1
    X = (X-np.min(X))/(np.max(X)-np.min(X))
    #normalize X to 0 mean and 1 variance
    X = sklearn.preprocessing.scale(X)
    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y).long()
    return X, Y


if __name__ == '__main__':
    df = pd.read_csv('methylation_table.csv')
    labels = pd.read_csv('labels.csv')
    X, y = load_dataset(df, labels)
     #Compute correlation between instances to get the edges
    correlation = torch.from_numpy(np.where(torch.tensor(np.corrcoef(X))>0.0, 1, 0))
    for i in range(correlation.shape[0]):
        correlation[i, i] = 0
    #make correlation a sparse matrix
    edge_index = torch.nonzero(correlation).t()
    data = torch_geometric.data.Data(x=X, edge_index=edge_index, y=y)
    #create train and test masks
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[:int(0.8*data.num_nodes)] = 1
    test_mask[int(0.8*data.num_nodes):] = 1
    data.train_mask = train_mask
    data.test_mask = test_mask

    #Instantiate the model, loss function and optimizer
    model = GCN(num_features=300, hidden_channels=100, num_classes=3)
    criterion = torch.nn.CrossEntropyLoss()
    lr = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc = 0
    #train
    model.train()
    for epoch in range(200):
        if epoch % 20 == 0:
            lr = lr * 0.5
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer.zero_grad()  # Clear gradients.
        out = model(data.x, data.edge_index)  # Perform a single forward pass.
        loss = criterion(out[data.train_mask],data.y[data.train_mask].reshape(-1))  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        model.eval()
        test_acc = test(model, data)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        print(f'Test Accuracy: {test_acc:.4f}')
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model, 'model.pt')
    
    #test
    model.eval()
    test_acc = test(model, data)
    print(f'Test Accuracy: {test_acc:.4f}')