import torch
import torch.nn as nn

class ProbabilisticLoss(nn.Module):
    def __init__(self, loss_weight=3.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, pred_mu, pred_logvar, target, mask=None):
        # Standard Gaussian negative log likelihood
        loss = 0.5 * (pred_logvar + ((target - pred_mu) ** 2) / torch.exp(pred_logvar))
        if mask is not None:
            loss = loss * mask
        return self.loss_weight * loss.mean()
