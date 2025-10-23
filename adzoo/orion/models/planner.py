import torch
import torch.nn as nn

class GenerativePlanner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, fused_embeds):
        return self.fc(fused_embeds)
