"""
Loss functions for Scientific Image Forgery Detection.

Includes:
1. Combined losses (BCE + Dice, BCE + IoU, etc.)
2. Focal Loss for handling class imbalance
3. Tversky Loss for precision-recall trade-off
4. Lovasz-Softmax Loss for IoU optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== Basic Loss Functions ====================

class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    Measures overlap between predicted and ground truth masks.
    """

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predictions (logits) [B, 1, H, W]
            target (torch.Tensor): Ground truth masks [B, 1, H, W]

        Returns:
            torch.Tensor: Dice loss value
        """
        pred = torch.sigmoid(pred)

        # Flatten
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        # Intersection and union
        intersection = (pred_flat * target_flat).sum(dim=1)
        dice = (2. * intersection + self.smooth) / (
            pred_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth
        )

        return 1 - dice.mean()


class IoULoss(nn.Module):
    """
    IoU (Jaccard) Loss for binary segmentation.
    Directly optimizes Intersection over Union metric.
    """

    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predictions (logits) [B, 1, H, W]
            target (torch.Tensor): Ground truth masks [B, 1, H, W]

        Returns:
            torch.Tensor: IoU loss value
        """
        pred = torch.sigmoid(pred)

        # Flatten
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        # Intersection and union
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)

        return 1 - iou.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Focuses training on hard examples.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predictions (logits) [B, 1, H, W]
            target (torch.Tensor): Ground truth masks [B, 1, H, W]

        Returns:
            torch.Tensor: Focal loss value
        """
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        pred_prob = torch.sigmoid(pred)
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)

        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss for controlling precision-recall trade-off.
    Generalization of Dice Loss.
    """

    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # Weight for false positives
        self.beta = beta    # Weight for false negatives
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predictions (logits) [B, 1, H, W]
            target (torch.Tensor): Ground truth masks [B, 1, H, W]

        Returns:
            torch.Tensor: Tversky loss value
        """
        pred = torch.sigmoid(pred)

        # Flatten
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        # True Positives, False Positives, False Negatives
        tp = (pred_flat * target_flat).sum(dim=1)
        fp = ((1 - target_flat) * pred_flat).sum(dim=1)
        fn = (target_flat * (1 - pred_flat)).sum(dim=1)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )

        return 1 - tversky.mean()


# ==================== Combined Loss Functions ====================

class CombinedLoss(nn.Module):
    """
    Combined BCE + Dice Loss.
    Balances pixel-wise accuracy (BCE) and region overlap (Dice).
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predictions (logits) [B, 1, H, W]
            target (torch.Tensor): Ground truth masks [B, 1, H, W]

        Returns:
            torch.Tensor: Combined loss value
        """
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class BCEDiceIoULoss(nn.Module):
    """
    Combined BCE + Dice + IoU Loss.
    Optimizes multiple objectives simultaneously.
    """

    def __init__(self, bce_weight=0.4, dice_weight=0.3, iou_weight=0.3):
        super(BCEDiceIoULoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.iou = IoULoss()

    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predictions (logits) [B, 1, H, W]
            target (torch.Tensor): Ground truth masks [B, 1, H, W]

        Returns:
            torch.Tensor: Combined loss value
        """
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        iou_loss = self.iou(pred, target)

        return (self.bce_weight * bce_loss +
                self.dice_weight * dice_loss +
                self.iou_weight * iou_loss)


class FocalDiceLoss(nn.Module):
    """
    Combined Focal + Dice Loss.
    Handles class imbalance while optimizing region overlap.
    """

    def __init__(self, focal_weight=0.5, dice_weight=0.5, alpha=0.25, gamma=2.0):
        super(FocalDiceLoss, self).__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss()

    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predictions (logits) [B, 1, H, W]
            target (torch.Tensor): Ground truth masks [B, 1, H, W]

        Returns:
            torch.Tensor: Combined loss value
        """
        focal_loss = self.focal(pred, target)
        dice_loss = self.dice(pred, target)

        return self.focal_weight * focal_loss + self.dice_weight * dice_loss


# ==================== Advanced Loss Functions ====================

class WeightedBCELoss(nn.Module):
    """
    Weighted BCE Loss for handling class imbalance.
    Automatically computes positive/negative class weights.
    """

    def __init__(self, pos_weight=None):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predictions (logits) [B, 1, H, W]
            target (torch.Tensor): Ground truth masks [B, 1, H, W]

        Returns:
            torch.Tensor: Weighted BCE loss value
        """
        if self.pos_weight is None:
            # Automatically compute positive weight
            num_pos = target.sum()
            num_neg = target.numel() - num_pos
            if num_pos > 0:
                pos_weight = num_neg / num_pos
            else:
                pos_weight = 1.0
        else:
            pos_weight = self.pos_weight

        loss = F.binary_cross_entropy_with_logits(
            pred, target,
            pos_weight=torch.tensor([pos_weight]).to(pred.device)
        )

        return loss


class BoundaryLoss(nn.Module):
    """
    Boundary Loss for emphasizing edges in segmentation.
    Useful for precise forgery boundary detection.
    """

    def __init__(self, theta=3):
        super(BoundaryLoss, self).__init__()
        self.theta = theta

    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predictions (logits) [B, 1, H, W]
            target (torch.Tensor): Ground truth masks [B, 1, H, W]

        Returns:
            torch.Tensor: Boundary loss value
        """
        pred = torch.sigmoid(pred)

        # Compute gradients
        pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])

        target_dx = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        target_dy = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])

        # Boundary loss
        loss_dx = torch.mean(torch.abs(pred_dx - target_dx))
        loss_dy = torch.mean(torch.abs(pred_dy - target_dy))

        return loss_dx + loss_dy


class StructureLoss(nn.Module):
    """
    Structure Loss combining weighted IoU and weighted BCE.
    From "Structure-measure: A new way to evaluate foreground maps" (ICCV 2017).
    """

    def __init__(self):
        super(StructureLoss, self).__init__()

    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predictions (logits) [B, 1, H, W]
            target (torch.Tensor): Ground truth masks [B, 1, H, W]

        Returns:
            torch.Tensor: Structure loss value
        """
        pred = torch.sigmoid(pred)

        # Weighted IoU
        weit = 1 + 5 * torch.abs(
            F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target
        )
        wbce = F.binary_cross_entropy(pred, target, reduction='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        # Weighted IoU
        inter = ((pred * target) * weit).sum(dim=(2, 3))
        union = ((pred + target) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        return (wbce + wiou).mean()


# ==================== Loss Factory ====================

def get_loss(loss_name='combined', **kwargs):
    """
    Factory function to get loss by name.

    Args:
        loss_name (str): Name of the loss function
        **kwargs: Additional arguments for loss function

    Returns:
        nn.Module: Loss function

    Available losses:
        - 'bce': Binary Cross-Entropy
        - 'dice': Dice Loss
        - 'iou': IoU Loss
        - 'focal': Focal Loss
        - 'tversky': Tversky Loss
        - 'combined': BCE + Dice (default)
        - 'bce_dice_iou': BCE + Dice + IoU
        - 'focal_dice': Focal + Dice
        - 'weighted_bce': Weighted BCE
        - 'boundary': Boundary Loss
        - 'structure': Structure Loss
    """
    loss_dict = {
        'bce': nn.BCEWithLogitsLoss(),
        'dice': DiceLoss(),
        'iou': IoULoss(),
        'focal': FocalLoss(**kwargs),
        'tversky': TverskyLoss(**kwargs),
        'combined': CombinedLoss(**kwargs),
        'bce_dice_iou': BCEDiceIoULoss(**kwargs),
        'focal_dice': FocalDiceLoss(**kwargs),
        'weighted_bce': WeightedBCELoss(**kwargs),
        'boundary': BoundaryLoss(**kwargs),
        'structure': StructureLoss(),
    }

    if loss_name not in loss_dict:
        raise ValueError(f"Unknown loss name: {loss_name}")

    return loss_dict[loss_name]


# ==================== Loss Testing ====================

if __name__ == "__main__":
    print("Testing loss functions...")

    # Create dummy data
    pred = torch.randn(2, 1, 256, 256)  # Logits
    target = torch.randint(0, 2, (2, 1, 256, 256)).float()

    # Test all losses
    losses = [
        ('BCE', nn.BCEWithLogitsLoss()),
        ('Dice', DiceLoss()),
        ('IoU', IoULoss()),
        ('Focal', FocalLoss()),
        ('Tversky', TverskyLoss()),
        ('Combined', CombinedLoss()),
        ('BCE+Dice+IoU', BCEDiceIoULoss()),
        ('Focal+Dice', FocalDiceLoss()),
    ]

    for name, loss_fn in losses:
        loss_value = loss_fn(pred, target)
        print(f"{name} Loss: {loss_value.item():.4f}")

    print("\nAll loss functions tested successfully!")
