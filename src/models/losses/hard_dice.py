import torch
import torch.nn as nn
from torchmetrics import Dice
from torchmetrics.classification import ConfusionMatrix


class HardDice(nn.Module):
    def __init__(self, average='micro', ignore_index=0):
        """
            Wrapper around torch metrics which has one goal: when both target and predicted are
            empty, we actually did what we were supposed to, and so we return 1 instead of 0
        """
        super().__init__()
        self.dice = Dice(average='micro', ignore_index=0)

    def forward(self, y_hat, y):
        assert y_hat.shape == y.shape

        if (y_hat == y).all():
            return torch.tensor(1.)
        else:
            return self.dice(y_hat, y)


def simple_dice(y_hat, y, ignore_index=0):
    y_hat = y_hat.cpu()
    y = y.cpu()

    cm = ConfusionMatrix(task='multiclass', num_classes=3)
    confusion_matrix = cm(y_hat, y)

    TP = 0
    FP = 0
    FN = 0

    for label in range(confusion_matrix.shape[0]):
        if label == ignore_index:
            continue

        tp = confusion_matrix[label, label]
        fp = torch.sum(confusion_matrix[:, label]) - tp
        fn = torch.sum(confusion_matrix[label, :]) - tp

        TP += tp.item()
        FP += fp.item()
        FN += fn.item()

    if TP+FP > 0:
        precision = TP/(TP+FP)
    else:
        precision = 1.

    if TP+FN > 0:
        recall = TP/(TP+FN)
    else:
        recall = 1.

    if TP+FP+FN > 0:
        dice = 2*TP / (2*TP + FP + FN)
    else:
        dice = 1.

    return dice, precision, recall
