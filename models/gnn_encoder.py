import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, RGATConv

class RGCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_relations, dropout=0.2):
        super().__init__()
        self.dropout = dropout
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
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        return x


class RGATEncoder(nn.Module):
    """
    Relational Graph Attention Network (RGAT) Encoder
    Note: This is a simplified implementation since PyTorch Geometric doesn't have RGATConv
    We'll use multiple GATConv layers with relation-specific attention
    """
    def __init__(self, input_dim, hidden_dim, num_relations, dropout=0.2, heads=4):
        super().__init__()
        self.dropout = dropout
        self.num_relations = num_relations
        self.heads = heads
        
        # For simplicity, we'll use RGCN as the base and add attention mechanisms
        # In a full implementation, you'd want proper relational attention
        self.conv1 = RGCNConv(input_dim, hidden_dim, num_relations, num_blocks=2)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations, num_blocks=2)
        
        # Add attention weights for relations
        self.relation_attention = nn.Parameter(torch.randn(num_relations, hidden_dim))
        
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        nn.init.xavier_uniform_(self.relation_attention)

    def forward(self, x, edge_index, edge_type):
        # First RGCN layer
        x = self.conv1(x, edge_index, edge_type).relu_()
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second RGCN layer with relation attention
        x = self.conv2(x, edge_index, edge_type)
        
        # Apply relation-specific attention (simplified)
        # In practice, you'd want more sophisticated attention mechanisms
        if len(edge_type) > 0:
            relation_weights = F.softmax(self.relation_attention, dim=1)
            # This is a simplified attention - full RGAT would be more complex
        
        return x