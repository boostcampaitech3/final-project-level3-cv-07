
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
mean_weight_data = {
    
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
        
class QuadraticKappaLoss(nn.Module):
    """
    Implements the Quadratic Weighted Kappa Loss Function.
    
    This loss was introduced in the
    "Weighted kappa loss function for multi-class classification
    of ordinal data in deep learning"
    
    weighted kappa is widely used in ordinal classification problems.
    
    The loss value lies in $ [-\infty, \log 2] $, where $ \log 2 $
    means the random prediction.
    
    and This implements is based on tfa.losses.WeightedKappaLoss
    """
    def __init__(self, 
                 num_classes : int,
                 weightage: Optional[str] = "quadratic",
                 name: Optional[str] = "cohen_kappa_loss",
                 epsilon: Optional[int] = 1e-6,
                 reduction: str = None,
                 device = 'cuda'
                 ):
        super().__init__()
        
        self.weightage = weightage
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.device = device
        
        label_vec = torch.arange(num_classes, dtype=torch.float64, device=self.device)
        self.row_label_vec = torch.reshape(label_vec, (1, num_classes)).to(torch.float64)
        self.col_label_vec = torch.reshape(label_vec, (num_classes, 1)).to(torch.float64)
        
        row_mat = self.row_label_vec.repeat((num_classes, 1))
        col_mat = self.col_label_vec.repeat((1, num_classes))
        
        if weightage == 'linear':
            self.weight_mat = torch.abs(col_mat - row_mat)
        else:
            self.weight_mat = (col_mat - row_mat) ** 2
    
    def forward(self, input, target):
        batch_size = input.size()[0]
        target : torch.Tensor = F.one_hot(target, num_classes=self.num_classes).to(torch.float64)
        cat_labels = torch.matmul(target, self.col_label_vec)
        cat_label_mat = cat_labels.repeat((1, self.num_classes))
        row_label_mat = self.row_label_vec.repeat((batch_size, 1))
        
        if self.weightage == 'linear':
            weight = torch.abs(cat_label_mat - row_label_mat)
        else:
            weight = (cat_label_mat - row_label_mat) ** 2
            
        numerator = torch.sum(weight * input)
        label_dist = torch.sum(target, dim=0, keepdim=True, dtype=torch.float64)
        pred_dist = torch.sum(input, dim=0, keepdim=True, dtype=torch.float64)
        w_pred_dist = torch.matmul(self.weight_mat, pred_dist.permute(1, 0))
        denominator = torch.sum(torch.matmul(label_dist, w_pred_dist), dtype=torch.float64)
        denominator /= batch_size
        
        loss = numerator / denominator
        loss = torch.where(torch.isnan(loss), 0.0, loss)
        return torch.log(loss + self.epsilon)
    
class F1_Loss(nn.Module):
    def __init__(self, num_classes, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
        self.num_classes = num_classes
        
    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.num_classes).to(torch.float32)
        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2* (precision*recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        return 1 - f1.mean()