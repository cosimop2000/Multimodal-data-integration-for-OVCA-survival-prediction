import pandas as pd
import torch
import models
import utilities


if __name__ == '__main__':
    df = pd.read_csv('methylation_table.csv')
    labels = pd.read_csv('labels.csv')
    X, y = utilities.load_dataset(df, labels)
    #To use the large model, use next line instead
    # X, y = utilities.load_dataset(df, labels, pca_components=0)
    #To perform regression, use next line instead
    # X, y = utilities.load_dataset(df, labels, task='regression')
    data = utilities.get_graph(X, y, train_size=0.8, correlation_threshold=0.85)
    #Instantiate the model, loss function and optimizer
    # model = models.GCNlarge(X.shape[1], 3)
    # model = models.GATsmall(num_features=300, hidden_channels=50, num_classes=3)
    model = models.GCNsmall(num_features=300, hidden_channels=100, num_classes=3)
    #For regression use this:
    # model = models.GCNsmall(num_features=300, hidden_channels=100, num_classes=1)
    criterion = torch.nn.CrossEntropyLoss()
    lr = 0.01 #Should be big for refression, like 1
    step = 20 
    rate = 0.5
    epochs = 18 #Should be big for regression, like 100, same for the big model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc = 0
    task = 'classification'
    #train
    model.train()
    for epoch in range(18):
        if epoch % step == 0:
            lr = lr * rate
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer.zero_grad()  # Clear gradients.
        out = model(data.x, data.edge_index)  # Perform a single forward pass.
        loss = criterion(out[data.train_mask],data.y[data.train_mask].reshape(-1))  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        model.eval()
        test_acc = utilities.test_GNN(model, data, verbose=True)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        print(f'Test Accuracy: {test_acc:.4f}')
        if task == 'classification':
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model, 'model.pt')
        elif task == 'regression':
            if test_acc < best_acc:
                best_acc = test_acc
                torch.save(model, 'model.pt')
    if task == 'classification':
        print(f'best Accuracy: {best_acc:.4f}')
    elif task == 'regression':
        print(f'best MAE: {best_acc:.4f}')