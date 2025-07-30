import torch
import torch.nn as nn

class DistMultDecoder(nn.Module):
    def __init__(self, num_relations, hidden_dim):
        super().__init__()
        self.rel_emb = nn.Parameter(torch.empty(num_relations, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.rel_emb)

    def forward(self, z, edge_index, edge_type):
        """
        Forward pass for DistMult scoring
        
        Args:
            z: Node embeddings [num_nodes, hidden_dim]
            edge_index: Edges to score [2, num_edges]
            edge_type: Edge types [num_edges]
        
        Returns:
            torch.Tensor: Link prediction scores [num_edges]
        """
        z_src, z_dst = z[edge_index[0]], z[edge_index[1]]
        rel = self.rel_emb[edge_type]
        # DistMult scoring: sum(h * r * t)
        scores = torch.sum(z_src * rel * z_dst, dim=1)
        return scores

    def get_score(self, head_embed, rel_embed, tail_embed):
        """Get score for specific embeddings"""
        return torch.sum(head_embed * rel_embed * tail_embed, dim=1)


class TransEDecoder(nn.Module):
    def __init__(self, num_relations, hidden_channels, p_norm=2):
        super().__init__()
        self.rel_emb = nn.Parameter(torch.empty(num_relations, hidden_channels))
        self.p_norm = p_norm  
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.rel_emb)

    def forward(self, z, edge_index, edge_type):
        """
        Forward pass for TransE scoring
        
        Args:
            z: Node embeddings [num_nodes, hidden_dim]
            edge_index: Edges to score [2, num_edges]
            edge_type: Edge types [num_edges]
        
        Returns:
            torch.Tensor: Link prediction scores [num_edges] (higher is better)
        """
        z_src, z_dst = z[edge_index[0]], z[edge_index[1]]  
        rel = self.rel_emb[edge_type]
        
        # TransE scoring: -||h + r - t||_p (negative distance, higher is better)
        scores = -((z_src + rel) - z_dst).norm(p=self.p_norm, dim=-1)
        return scores
    
    def get_score(self, head_embed, rel_embed, tail_embed):
        """Get score for specific embeddings"""
        return -((head_embed + rel_embed) - tail_embed).norm(p=self.p_norm, dim=-1)