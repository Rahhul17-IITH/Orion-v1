import torch
import torch.nn as nn

class QTFormer(nn.Module):
    def __init__(self, embed_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8) for _ in range(num_layers)]
        )
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
