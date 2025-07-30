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
        z_src, z_dst = z[edge_index[0]], z[edge_index[1]]
        rel = self.rel_emb[edge_type]
        return torch.sum(z_src * rel * z_dst, dim=1), z_src, rel, z_dst

    def get_score(self, head_embed, rel_embed, tail_embed):
        return torch.sum(head_embed * rel_embed * tail_embed, dim=1)


class TransEDecoder(torch.nn.Module):
    def __init__(self, num_relations, hidden_channels, p_norm=2):
        super().__init__()
        self.rel_emb = torch.nn.Parameter(torch.empty(num_relations, hidden_channels))
        self.p_norm = p_norm  
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.rel_emb)

    def forward(self, z, edge_index, edge_type):
        z_src, z_dst = z[edge_index[0]], z[edge_index[1]]  
        rel = self.rel_emb[edge_type]  # Relation embedding

        return -((z_src + rel) - z_dst).norm(p=self.p_norm, dim=-1) 