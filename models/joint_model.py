import torch
import torch.nn as nn
import torch.nn.functional as F

class JointModel(nn.Module):
    def __init__(self, gnn_encoder, decoder, classifier):
        super().__init__()
        self.gnn_encoder = gnn_encoder
        self.decoder = decoder
        self.classifier = classifier
        
    def forward(self, x, edge_index, edge_type, edge_label_index):
        # Shared GNN encoder
        z = self.gnn_encoder(x, edge_index, edge_type)
        
        # Link prediction (auxiliary task)
        link_logits, h_embed, r_embed, t_embed = self.decoder(
            z, 
            edge_label_index, 
            torch.zeros(edge_label_index.size(1), dtype=torch.long, device=edge_label_index.device)
        )
        
        # Classification (main task)
        donor_emb = z[edge_label_index[0]]
        recip_emb = z[edge_label_index[1]]
        pair_emb = torch.cat([donor_emb, recip_emb], dim=1)
        class_logits = self.classifier(pair_emb)
        
        return link_logits, class_logits, z