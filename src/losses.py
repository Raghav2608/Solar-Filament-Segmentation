"""
Loss functions & metrics for BraTS multi-label segmentation
============================================================
Three sub-region channels are treated independently as binary problems
(multi-label, not multi-class → sigmoid, not softmax).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
#  Dice loss
# ─────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary / multi-label segmentation.
    Expects raw logits; applies sigmoid internally.

    Args:
        smooth    : Laplace smoothing to avoid div-by-zero
        reduction : 'mean' (average over channels) or 'none'
    """

    def __init__(self, smooth: float = 1e-5, reduction: str = "mean"):
        super().__init__()
        self.smooth    = smooth
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)                           # (B, C, H, W, D)
        # Flatten spatial dims
        probs   = probs.contiguous().view(probs.shape[0], probs.shape[1], -1)
        targets = targets.contiguous().view(targets.shape[0], targets.shape[1], -1)

        intersection = (probs * targets).sum(-1)                # (B, C)
        union        = probs.sum(-1) + targets.sum(-1)          # (B, C)
        dice_per_ch  = (2 * intersection + self.smooth) / (union + self.smooth)
        loss_per_ch  = 1 - dice_per_ch                         # (B, C)

        if self.reduction == "mean":
            return loss_per_ch.mean()
        return loss_per_ch                                      # (B, C)


# ─────────────────────────────────────────────
#  Combined BCE + Dice loss
# ─────────────────────────────────────────────

class BCEDiceLoss(nn.Module):
    """
    Weighted combination of BCE and Dice loss.
    Both losses are computed on raw logits.

    Args:
        bce_weight  : contribution of BCE term (default 0.5)
        dice_weight : contribution of Dice term (default 0.5)
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.bce  = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.bce_weight  * self.bce(logits, targets)
            + self.dice_weight * self.dice(logits, targets)
        )


# ─────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────


@torch.no_grad()
def dice_metric(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """
    Hard Dice coefficient per channel, averaged over the batch.

    Returns:
        Tensor of shape (C,) — one Dice per sub-region.
    """
    probs   = (torch.sigmoid(logits) > threshold).float()
    targets = targets.float()

    probs   = probs.view(probs.shape[0], probs.shape[1], -1)
    targets = targets.view(targets.shape[0], targets.shape[1], -1)

    intersection = (probs * targets).sum(-1)                    # (B, C)
    union        = probs.sum(-1) + targets.sum(-1)              # (B, C)
    dice         = (2 * intersection + smooth) / (union + smooth)  # (B, C)
    return dice.mean(0)                                         # (C,)


@torch.no_grad()
def hausdorff95(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    95th-percentile Hausdorff distance (averaged over channels & batch).
    Requires scipy. Falls back gracefully if not installed.
    Operates on CPU numpy arrays.
    """
    try:
        from scipy.spatial.distance import directed_hausdorff
        import numpy as np

        pred_np   = (torch.sigmoid(pred) > threshold).cpu().numpy()
        target_np = target.cpu().numpy().astype(bool)

        hds = []
        for b in range(pred_np.shape[0]):
            for c in range(pred_np.shape[1]):
                p = np.argwhere(pred_np[b, c])
                t = np.argwhere(target_np[b, c])
                if len(p) == 0 or len(t) == 0:
                    continue
                d1 = directed_hausdorff(p, t)[0]
                d2 = directed_hausdorff(t, p)[0]
                hds.append(max(d1, d2))
        return float(np.percentile(hds, 95)) if hds else float("nan")
    except ImportError:
        return float("nan")


# ─────────────────────────────────────────────
#  Sanity check
# ─────────────────────────────────────────────

if __name__ == "__main__":
    B, C, H, W, D = 2, 3, 32, 32, 32
    logits  = torch.randn(B, C, H, W, D)
    targets = (torch.rand(B, C, H, W, D) > 0.8).float()

    criterion = BCEDiceLoss()
    loss = criterion(logits, targets)
    print(f"BCEDice loss: {loss.item():.4f}")

    dices = dice_metric(logits, targets)
    for d in dices:
        print(f"  Dice {d.item():.4f}")