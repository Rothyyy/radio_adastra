import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=256, out_dim=128, dropout_prob=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),    # Embedding size should be (B, 768)
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),        
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        out = self.head(x) 
        return out
