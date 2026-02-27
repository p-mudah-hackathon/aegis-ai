import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    # 
    #     Binary Focal Loss for fraud detection with extreme class imbalance.
    # 
    #     FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    # 
    #     Args:
    #         alpha (float): Weighting factor for positive class. Default 0.25.
    #         gamma (float): Focusing parameter. Higher = more focus on hard samples.
    #         reduction (str): 'mean', 'sum', or 'none'.
    #     

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # 
        #         Args:
        #             logits: Raw model outputs (before sigmoid), shape [N]
        #             targets: Binary labels, shape [N]
        #         
        probs = torch.sigmoid(logits)
        targets = targets.float()

        # p_t = p for positive, 1-p for negative
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal weight
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        # BCE loss (numerically stable)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        loss = focal_weight * bce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class WeightedBCELoss(nn.Module):
    # Weighted BCE with pos_weight for imbalance.

    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        if self.pos_weight is not None:
            pw = torch.tensor([self.pos_weight], device=logits.device)
        else:
            pw = None
        return F.binary_cross_entropy_with_logits(
            logits, targets.float(), pos_weight=pw)
