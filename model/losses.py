from typing import Optional, Sequence

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

class Derma_FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2., reduction='mean'):
        super().__init__()

        self.l_oil = FocalLoss(weight=weight, gamma=gamma, reduction=reduction)
        self.l_sen = FocalLoss(weight=weight, gamma=gamma, reduction=reduction)
        self.l_pig = FocalLoss(weight=weight, gamma=gamma, reduction=reduction)
        self.l_wri = FocalLoss(weight=weight, gamma=gamma, reduction=reduction)
        self.l_hyd = FocalLoss(weight=weight, gamma=gamma, reduction=reduction)
        
    def forward(self, input, targets):
        for i in range(len(targets)):
            targets[i] = torch.where(targets[i] < 0, torch.argmax(input[i], dim=-1), targets[i])
        
        loss_oil = self.l_oil(input[0], targets[0])
        loss_sen = self.l_sen(input[1], targets[1])
        loss_pig = self.l_pig(input[2], targets[2])
        loss_wri = self.l_wri(input[3], targets[3])
        loss_hyd = self.l_hyd(input[4], targets[4])
        
        total_loss = loss_oil + loss_sen + loss_pig + loss_wri + loss_hyd
        
        return total_loss, (loss_oil, loss_sen, loss_pig, loss_wri, loss_hyd)

class FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )