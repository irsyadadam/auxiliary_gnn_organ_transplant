import torch
import torch.nn as nn
import torch.nn.functional as F

class JointModel(nn.Module):
    def __init__(self, gnn_encoder, decoder, classifier):
        super().__init__()
        self.gnn_encoder = gnn_encoder
        self.decoder = decoder
        self.classifier = classifier
        
    def forward(self, x, edge_index, edge_type, donor_nodes=None, recipient_nodes=None, 
                link_edges=None, link_edge_types=None):
        """
        Forward pass for joint multi-task learning
        
        Args:
            x: Node features [num_nodes, feature_dim]
            edge_index: Full graph edges [2, num_edges]
            edge_type: Edge types for full graph [num_edges]
            donor_nodes: Donor node indices for classification [batch_size]
            recipient_nodes: Recipient node indices for classification [batch_size]
            link_edges: Edges for link prediction [2, num_link_edges]
            link_edge_types: Edge types for link prediction [num_link_edges]
        
        Returns:
            dict: Contains 'embeddings', 'link_logits', 'class_logits'
        """
        # Shared GNN encoder
        z = self.gnn_encoder(x, edge_index, edge_type)
        
        outputs = {'embeddings': z}
        
        # Link prediction (auxiliary task)
        if link_edges is not None and link_edge_types is not None:
            # Handle different decoder return formats
            decoder_output = self.decoder(z, link_edges, link_edge_types)
            if isinstance(decoder_output, tuple):
                link_logits = decoder_output[0]  # DistMult returns (scores, h, r, t)
            else:
                link_logits = decoder_output  # TransE returns scores directly
            outputs['link_logits'] = link_logits
        
        # Classification (main task)
        if donor_nodes is not None and recipient_nodes is not None:
            donor_emb = z[donor_nodes]
            recip_emb = z[recipient_nodes]
            # Use correct classifier interface
            class_logits = self.classifier(donor_emb, recip_emb)
            outputs['class_logits'] = class_logits
        
        return outputs
    
    def get_embeddings(self, x, edge_index, edge_type):
        """Get node embeddings only"""
        return self.gnn_encoder(x, edge_index, edge_type)
    
    def predict_links(self, embeddings, link_edges, link_edge_types):
        """Predict links using embeddings"""
        decoder_output = self.decoder(embeddings, link_edges, link_edge_types)
        if isinstance(decoder_output, tuple):
            return decoder_output[0]
        return decoder_output
    
    def predict_outcomes(self, embeddings, donor_nodes, recipient_nodes):
        """Predict outcomes using embeddings"""
        donor_emb = embeddings[donor_nodes]
        recip_emb = embeddings[recipient_nodes]
        return self.classifier(donor_emb, recip_emb)