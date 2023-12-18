import torch
import torch.nn as nn


class SoftDice(nn.Module):
    def __init__(self, alpha=0.5, smooth=1., eps=1e-8):
        """
        Soft Dice Loss. Loosely based on https://arxiv.org/pdf/1606.04797.pdf
        Currently only works for 2 foreground classes, but should be easy to extend.

        Arguments:
            alpha (int): weight given to first foreground class. alpha in [0,1]. Default 0.5.
            smooth (int): constant added to both numerator and denominator in dice calc. Default: 1.
            eps (int): we dont want to devide by zero. Default 1e-8.
        """
        super().__init__()

        assert alpha >= 0 and alpha <= 1
        assert smooth >= 0

        self.alpha = alpha
        self.smooth = smooth
        self.eps = eps

    def forward(self, p, y):
        # p dim: (B, C, H, W)
        # y dim: (B, 1, H, W)

        assert p.shape[0] == y.shape[0]
        assert p.shape[1] == 3
        assert y.shape[1] == 1
        assert p.shape[2] == y.shape[2]
        assert p.shape[3] == y.shape[3]

        # (B, C, H, W)
        y_onehot = torch.zeros(p.shape, device=p.device).scatter(1, y.long(), 1)

        image_dims = (2, 3)  # dimensions which are not batch (0) or category (1)

        # tp, fp, fn dims (B, C)
        tp = torch.sum(p * y_onehot, dim=image_dims)
        fp = torch.sum(p * (1 - y_onehot), dim=image_dims)
        fn = torch.sum((1 - p) * y_onehot, dim=image_dims)

        class_dice = (2*tp + self.smooth) / (2*tp + fp + fn + self.smooth + self.eps)  # (B, C)

        # perform weighted average between two foreground classes.
        w = torch.tensor([0, self.alpha, 1-self.alpha], device=p.device)
        dice = class_dice@w  # (B)

        # we reduce using mean over batch dim. This could also be e.g. a sum.
        return torch.mean(dice)
