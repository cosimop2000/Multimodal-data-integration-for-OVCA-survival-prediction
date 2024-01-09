import torch
import torch_geometric
from torch.nn import functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv

class GCNsmall(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.gat = torch_geometric.nn.GATConv(hidden_channels, hidden_channels, heads=8, dropout=0.6)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
class GCNlarge(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        h1 = 2000
        h2 = 500
        h3 = 100
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(num_features, h1)
        self.conv2 = GCNConv(h1, h2)
        self.conv3 = GCNConv(h2, h3)
        self.conv4 = GCNConv(h3, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv4(x, edge_index)
        return x

class GATsmall(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, heads=8, dropout=0.6):
        super().__init__()
        torch.manual_seed(1234567)
        self.gat1 = GATConv(num_features, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATConv(heads*hidden_channels, int(hidden_channels*0.5), heads=8, dropout=dropout)
        self.conv2 = GCNConv(int(0.5*heads*hidden_channels), num_classes)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = x.relu()
        x = self.gat2(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x