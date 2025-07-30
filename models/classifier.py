import torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        """
        Multi-layer perceptron classifier for binary classification
        
        Args:
            input_dim: Input feature dimension (e.g., concatenated embeddings)
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (1 for binary classification)
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization"""
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, donor_embed, recipient_embed):
        """
        Forward pass for outcome prediction
        
        Args:
            donor_embed: Donor node embeddings [batch_size, embed_dim]
            recipient_embed: Recipient node embeddings [batch_size, embed_dim]
            
        Returns:
            torch.Tensor: Predictions [batch_size, output_dim]
        """
        # Concatenate donor and recipient embeddings
        pair_embed = torch.cat([donor_embed, recipient_embed], dim=1)
        return self.net(pair_embed)