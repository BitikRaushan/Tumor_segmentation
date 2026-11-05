import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.boundary_loss import BoundaryLoss

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        num = 2.0 * torch.sum(probs * target) + self.smooth
        den = torch.sum(probs) + torch.sum(target) + self.smooth
        return 1.0 - (num / den)

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.8, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        p_t = probs * target + (1 - probs) * (1 - target)
        loss = bce * ((1 - p_t) ** self.gamma)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_t * loss
        return loss.mean()

class CombinedLoss(nn.Module):
    def __init__(self, l_focal=1.0, l_dice=1.0, l_bdry=0.5):
        super().__init__()
        self.focal = BinaryFocalLoss()
        self.dice = SoftDiceLoss()
        self.boundary = BoundaryLoss()
        self.l_focal = l_focal
        self.l_dice = l_dice
        self.l_bdry = l_bdry

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        f_loss = self.focal(logits, target)
        d_loss = self.dice(logits, target)
        b_loss = self.boundary(logits, target)
        return self.l_focal * f_loss + self.l_dice * d_loss + self.l_bdry * b_loss