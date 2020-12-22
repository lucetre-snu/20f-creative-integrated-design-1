import torch
import numpy as np


# output(-inf ~ +inf) target(0 or 1)
def f1_loss(output, target):
    eps = torch.finfo(torch.float64).eps
    y_pred = torch.sigmoid(output)
    y_true = target

    tp = torch.sum(y_true*y_pred)
    tn = torch.sum((1-y_true)*(1-y_pred))
    fp = torch.sum((1-y_true)*y_pred)
    fn = torch.sum(y_true*(1-y_pred))

    p = (tp) / (tp + fp + eps)
    r = (tp) / (tp + fn + eps)
    f1 = 2*p*r / (p+r + eps)
    f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
    return 1 - torch.mean(f1)

class weighted_loss_and_f1_loss:
    def __init__(self, weighted_loss):
        self.weighted_loss = weighted_loss

    def loss(self, output, target):
        loss = 0.5 * f1_loss(output, target)
        loss += 0.5 * self.weighted_loss(output, target)
        return loss
