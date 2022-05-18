from h11 import Data
from data.dataset import Derma_dataset
from model.model import *
from model.losses import FocalLoss
from model.metric import ArcMarginProduct
from torch.utils.data.dataloader import DataLoader
import os
import torch

import torch.nn.functional as F
import numpy as np
import random

def set_seed(seed : int) -> None:
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name

BATCH_SIZE = 16

device = torch.device('cuda')

train_dataset = Derma_dataset('../data/train', transform=None)
val_dataset = Derma_dataset('../data/val', transform=None)

train_dataloader = DataLoader(train_dataset, 
                              BATCH_SIZE, 
                              shuffle=True,
                              num_workers=4)

val_dataloader = DataLoader(val_dataset,
                            BATCH_SIZE,
                            num_workers=4)

