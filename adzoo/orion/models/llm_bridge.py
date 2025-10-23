import torch
import torch.nn as nn

class LLMBridge(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, vision_embeds, text_embeds):
        x = torch.cat([vision_embeds, text_embeds], dim=-1)
        return self.fc(x)
