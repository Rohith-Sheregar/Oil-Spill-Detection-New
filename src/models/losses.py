"""
BCE + Dice loss with binary label smoothing (eps=0.1), per the synopsis.

Implementation choice worth flagging: label smoothing is applied to the BCE
term only, not to the Dice term. Smoothing the targets fed into Dice would
distort the overlap metric itself (Dice is already an IoU-like ratio;
blurring both numerator and denominator toward 0.5 changes what the loss is
actually optimizing in a way that BCE's smoothing doesn't). Dice is computed
against the unsmoothed mask.
"""
import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, smooth=1.0, label_smoothing=0.1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.eps = label_smoothing

    def forward(self, logits, targets):
        targets_smooth = targets * (1 - self.eps) + 0.5 * self.eps
        bce_loss = self.bce(logits, targets_smooth)

        probs = torch.sigmoid(logits).flatten()
        t = targets.flatten()
        intersection = (probs * t).sum()
        dice_loss = 1 - (2 * intersection + self.smooth) / (probs.sum() + t.sum() + self.smooth)

        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


if __name__ == "__main__":
    loss_fn = BCEDiceLoss()
    logits = torch.randn(2, 1, 64, 64)
    targets = (torch.rand(2, 1, 64, 64) > 0.7).float()
    print(loss_fn(logits, targets).item())
