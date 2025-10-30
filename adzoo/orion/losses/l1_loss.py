import torch
import torch.nn as nn

class L1Loss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target, mask=None):
        if mask is not None:
            loss = torch.abs((pred - target) * mask).mean()
        else:
            loss = torch.abs(pred - target).mean()
        return self.loss_weight * loss
