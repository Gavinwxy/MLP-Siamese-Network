import torch
import torch.nn as nn


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, metric, margin=2.5):
        super(ContrastiveLoss, self).__init__()
        self.metric = metric
        self.margin = margin

    def forward(self, output1, output2, label):
        distance = self.metric(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))

        return loss_contrastive

class TripletLoss(torch.nn.Module):
    """
    Based on: https://arxiv.org/pdf/1503.03832.pdf
    """

    def __init__(self, metric, margin=45):
        super(TripletLoss, self).__init__()
        self.metric = metric
        self.margin = margin

    def forward(self, output1, output2, output3):
        distance1 = self.metric(output1, output2)
        distance2 = self.metric(output1, output3)
        loss_triplet = torch.sum(torch.clamp(distance1.pow(2) - distance2.pow(2) + self.margin, min=0.0))

        return loss_triplet

class LogisticLoss(nn.CrossEntropyLoss):
    def __init__(self):
        nn.CrossEntropyLoss.__init__(self)
    
class CosFace(nn.CrossEntropyLoss):
    def __init__(self):
        nn.CrossEntropyLoss.__init__(self)

class ArcFace(nn.CrossEntropyLoss):
    def __init__(self):
        nn.CrossEntropyLoss.__init__(self)
