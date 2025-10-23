from .models.qtformer import QTFormer
from .models.llm_bridge import LLMBridge
from .models.planner import GenerativePlanner
import torch.nn as nn

class OrionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.qtformer = QTFormer(embed_dim=256, num_layers=6)
        self.llm_bridge = LLMBridge(input_dim=512, hidden_dim=256)
        self.planner = GenerativePlanner(input_dim=256, output_dim=20)  # 20: number of trajectory points

    def forward(self, rgb_sequence, text_input):
        qt_features = self.qtformer(rgb_sequence)
        fused = self.llm_bridge(qt_features, text_input)
        action = self.planner(fused)
        return action
