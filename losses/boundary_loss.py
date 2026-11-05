"""
Boundary Distance Regularization Loss to eliminate the paper's 82 mm HD95 error.
Uses Euclidean Distance Transforms (EDT) on ground truth segmentations.
"""
import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import distance_transform_edt

class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def compute_edt(self, mask: np.ndarray) -> np.ndarray:
        """Computes signed Euclidean distance transform of binary ground truth."""
        posmask = mask.astype(bool)
        if not posmask.any():
            return np.zeros_like(mask, dtype=np.float32)
        
        negmask = ~posmask
        dist_out = distance_transform_edt(negmask)
        dist_in = distance_transform_edt(posmask)
        
        # Signed distance: positive outside, negative inside
        signed_dist = (dist_out - 1) * negmask - (dist_in - 1) * posmask
        return signed_dist.astype(np.float32)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predicted tensor [B, 1, H, W]
            target: Binary ground truth [B, 1, H, W]
        """
        probs = torch.sigmoid(logits)
        b, _, _, _ = target.shape
        
        # Compute distance maps dynamically or retrieve from precomputed cache
        target_np = target.detach().cpu().numpy()
        dist_maps = np.zeros_like(target_np)
        for i in range(b):
            dist_maps[i, 0] = self.compute_edt(target_np[i, 0])
            
        dist_tensor = torch.from_numpy(dist_maps).to(logits.device)
        # Penalize probability mass placed in regions with high positive distance
        boundary_loss = torch.mean(probs * dist_tensor)
        return boundary_loss