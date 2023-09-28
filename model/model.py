import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, GATConv, SAGEConv, GCNConv, GraphConv
from torch_geometric.nn.inits import reset
from torch_scatter import scatter


class FeatureBooster(nn.Module):
    def __init__(self, dim, slices):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim //slices, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim//slices, dim, bias=False),
        )
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.mlp)

    def forward(self, x, size=None):
        batch = torch.tensor(range(x.shape[0]))
        max_result = scatter(x, batch, dim=0, dim_size=size, reduce='max')
        sum_result = scatter(x, batch, dim=0, dim_size=size, reduce='sum')
        max_out = self.mlp(max_result)
        sum_out = self.mlp(sum_result)
        y = torch.sigmoid(max_out + sum_out)
        y = y[batch]
        return x * y


class Net(torch.nn.Module):
    def __init__(self, dim_in=165, dim_hidden=128, slices=2, num_layer=1, f_att=True):
        super(Net, self).__init__()
        self.num_layer = num_layer
        self.conv1 = SAGEConv(dim_in, dim_hidden)
        self.conv2 = GATConv(dim_hidden,dim_hidden//2)
        self.conv3 = ChebConv(dim_hidden//2, 1, K=1)

        self.f_att = f_att
        if self.f_att:
            assert num_layer >= 1
            self.f_att = FeatureBooster(dim_in, slices)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.f_att:
            for layer in range(self.num_layer):
                x = self.f_att(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv3(x, edge_index)
        return torch.sigmoid(x)

class GCNNet(torch.nn.Module):
    def __init__(self, dim_in=165, dim_hidden=128, slices=2, num_layer=1, f_att=True):
        super(GCNNet, self).__init__()
        self.num_layer = num_layer
        self.conv1 = GCNConv(dim_in, dim_hidden)
        self.conv2 = GCNConv(dim_hidden,dim_hidden//2)
        self.conv3 = GCNConv(dim_hidden//2, 1)

        self.f_att = f_att
        if self.f_att:
            assert num_layer >= 1
            self.f_att = FeatureBooster(dim_in, slices)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.f_att:
            for layer in range(self.num_layer):
                x = self.f_att(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv3(x, edge_index)
        return torch.sigmoid(x)

class GATNet(torch.nn.Module):
    def __init__(self, dim_in=165, dim_hidden=128, slices=2, num_layer=1, f_att=True):
        super(GATNet, self).__init__()
        self.num_layer = num_layer
        self.conv1 = GATConv(dim_in, dim_hidden)
        self.conv2 = GATConv(dim_hidden,dim_hidden//2)
        self.conv3 = GATConv(dim_hidden//2, 1)

        self.f_att = f_att
        if self.f_att:
            assert num_layer >= 1
            self.f_att = FeatureBooster(dim_in, slices)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.f_att:
            for layer in range(self.num_layer):
                x = self.f_att(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv3(x, edge_index)
        return torch.sigmoid(x)

class SimpleNet(torch.nn.Module):
    def __init__(self, dim_in=165, dim_hidden=128):
        super(SimpleNet, self).__init__()
        self.conv1 = GraphConv(dim_in, dim_hidden)
        self.conv2 = GraphConv(dim_hidden, 1)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)
        
class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

