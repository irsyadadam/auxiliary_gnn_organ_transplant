import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class RGCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_relations):
        super().__init__()
        self.conv1 = RGCNConv(input_dim, hidden_dim, num_relations,
                              num_blocks=2)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations,
                              num_blocks=2)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type).relu_()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        return x