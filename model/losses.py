from numbers import Rational
from typing import Optional, Sequence

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

weight_data = {
    0 : {'oil' : torch.tensor([75.5, 3.25, 1.90, 7.0, 104.46]), 'sensitive' : torch.tensor([2.9, 2.1, 6.345, 71.47, 226.4]), 
         'pigmentation' : torch.tensor([4.63, 2.24, 4.04, 13.85, 50.29])},
    1 : {'oil' : torch.tensor([134.4, 4.36, 1.78, 5.6, 42.0]), 'sensitive' : torch.tensor([3.13, 1.8, 10.0, 55.91, 167.75]), 
         'wrinkle' : torch.tensor([2.34, 2.7, 5.55, 55.91, 167.75])},
    2 : {'oil' : torch.tensor([226.7, 4.15, 1.67, 6.73, 113.4]), 'sensitive' : torch.tensor([2.32, 2.30, 8.29, 85.0, 340.0]), 
         'pigmentation' : torch.tensor([4.78, 2.34, 3.69, 15.1, 35.78]), 'wrinkle' : torch.tensor([8.5, 1.98, 3.17, 19.42, 85.0])},
    3 : {'sensitive' : torch.tensor([1.76, 2.93, 11.92, 170.0, 340.0]), 'wrinkle' : torch.tensor([1.82, 3.15, 9.32, 40.05, 681.0]), 
         'hydration' : torch.tensor([16.97, 3.73, 2.43, 5.10, 15.08])}
}

class Derma_FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2., reduction='mean', ignore_index=-100, part=None):
        super().__init__()
        
        if type(part) != int:
            raise TypeError('Please check part type. your type(part) is {}, but need int type'.format(type(part)))
        self.loss_dict = self.build_criterion(weight, gamma, reduction, ignore_index, part)

    def build_criterion(self, weight=None, gamma=2, reduction='mean', ignore_index=-100, part=None):
        if part < 0 or part > 3:
            raise ValueError('part value is in 0 ~ 3. but input value is {part}')
        elif part == 0:
            print('building cheeck classification loss...')
            part_cat = ['oil', 'sensitive', 'pigmentation']
            loss_dict = nn.ModuleDict()
            for cat in part_cat:
                cat_weight = weight_data[part][cat].to('cuda')
                loss = FocalLoss(cat_weight, gamma, reduction, ignore_index)
                loss_dict[cat] = loss
        elif part == 1:
            print('building upper_face classification loss...')
            part_cat = ['oil', 'sensitive', 'wrinkle']
            loss_dict = nn.ModuleDict()
            for cat in part_cat:
                cat_weight = weight_data[part][cat].to('cuda')
                loss = FocalLoss(cat_weight, gamma, reduction, ignore_index)
                loss_dict[cat] = loss
        elif part == 2:
            print('building mid_face classification loss...')
            part_cat = ['oil', 'sensitive', 'pigmentation', 'wrinkle']
            loss_dict = nn.ModuleDict()
            for cat in part_cat:
                cat_weight = weight_data[part][cat].to('cuda')
                loss = FocalLoss(cat_weight, gamma, reduction, ignore_index)
                loss_dict[cat] = loss
        else:
            print('building lower_face classification loss...')
            part_cat = ['sensitive', 'wrinkle', 'hydration']
            loss_dict = nn.ModuleDict()
            for cat in part_cat:
                cat_weight = weight_data[part][cat].to('cuda')
                loss = FocalLoss(cat_weight, gamma, reduction, ignore_index)
                loss_dict[cat] = loss
        
        return loss_dict
    
    def forward(self, input, targets):
        total_loss = 0
        
        losses_dict = {}
        for cat in self.loss_dict.keys():
            loss = self.loss_dict[cat](input[cat], targets[cat])
            total_loss += loss
            losses_dict[cat] = loss
        
        return total_loss, losses_dict

class Derma_CELoss(nn.Module):
    def __init__(self, weight=None, reduction='mean', ignore_index=-100, part=None):
        super().__init__()
        if type(part) != int:
            raise TypeError('Please check part type. your type(part) is {}, but need int type'.format(type(part)))
        self.loss_dict = self.build_criterion(weight, reduction, ignore_index, part)

    def build_criterion(self, weight, reduction, ignore_index, part):
        if part < 0 or part > 3:
            raise ValueError('part value is in 0 ~ 3. but input value is {part}')
        elif part == 0:
            print('building cheeck classification loss...')
            part_cat = ['oil', 'sensitive', 'pigmentation']
            loss_dict = nn.ModuleDict()
            for cat in part_cat:
                loss = nn.CrossEntropyLoss(weight, reduction, ignore_index)
                loss_dict[cat] = loss
        elif part == 1:
            print('building upper_face classification loss...')
            part_cat = ['oil', 'sensitive', 'wrinkle']
            loss_dict = nn.ModuleDict()
            for cat in part_cat:
                loss = nn.CrossEntropyLoss(weight, reduction, ignore_index)
                loss_dict[cat] = loss
        elif part == 2:
            print('building mid_face classification loss...')
            part_cat = ['oil', 'sensitive', 'pigmentation', 'wrinkle']
            loss_dict = nn.ModuleDict()
            for cat in part_cat:
                loss = nn.CrossEntropyLoss(weight, reduction, ignore_index)
                loss_dict[cat] = loss
        else:
            print('building lower_face classification loss...')
            part_cat = ['sensitive', 'wrinkle', 'hydration']
            loss_dict = nn.ModuleDict()
            for cat in part_cat:
                loss = nn.CrossEntropyLoss(weight, reduction, ignore_index)
                loss_dict[cat] = loss
        
        return loss_dict
    
    def forward(self, input, targets):
        total_loss = 0
        
        losses_dict = {}
        for cat in self.loss_dict.keys():
            loss = self.loss_dict[cat](input[cat], targets[cat])
            total_loss += loss
            losses_dict[cat] = loss
        
        return total_loss, losses_dict

class FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., reduction='mean', ignore_index=-100):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction,
            ignore_index=self.ignore_index
        )