import torch
import torch.nn as nn


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, metric, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.metric = metric
        self.margin = margin

    def forward(self, output1, output2, label):
        distance = self.metric(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))

        return loss_contrastive

class ChopraLoss(torch.nn.Module):
    """
    ChopraLoss loss function.
    Based on: https://ieeexplore.ieee.org/abstract/document/1467314
    """

    def __init__(self, metric, Q=2.0):
        super(ChopraLoss, self).__init__()
        self.metric = metric
        self.Q = Q

    def forward(self, output1, output2, label):
        distance = self.metric(output1, output2)
        loss_chopra = torch.mean((1 - label) * (2 / self.Q) * torch.pow(distance, 2) +
                                 (label) * (2 * self.Q) * torch.exp((-2.77 / self.Q) * distance))

        return loss_chopra
