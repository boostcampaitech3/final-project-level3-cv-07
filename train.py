from email.quoprimime import header_decode
from data.dataset import Derma_dataset
from model.model import Convnext_custom
from model.losses import FocalLoss, Derma_FocalLoss
from model.metric import ArcMarginProduct
from torch.utils.data import DataLoader
from tqdm import tqdm
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

BATCH_SIZE = 6
NUM_CLASSES = 5
EPOCH = 50

device = torch.device('cuda')

train_dataset = Derma_dataset('/opt/ml/input/data/train', transform=None)
val_dataset = Derma_dataset('/opt/ml/input/data/val', transform=None)

train_dataloader = DataLoader(train_dataset, 
                              batch_size = BATCH_SIZE, 
                              shuffle=True,
                              num_workers=4)

val_dataloader = DataLoader(val_dataset,
                            batch_size = BATCH_SIZE,
                            num_workers=4)


model = Convnext_custom('small')

criterion = Derma_FocalLoss(gamma=2).to(device)

# metric_fc = ArcMarginProduct(model.get_last_dim(), NUM_CLASSES)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(),
                              lr=0.0001, weight_decay=0.05)


for i in range(EPOCH):
    model.train()
    for idx, data in tqdm(enumerate(train_dataloader)):
        X, Ys = data
        X = X.to(device)
        
        pred_list = model(X)
        label_list = [Ys[cat].to(device) for cat in Ys.keys()]
        
        batch_loss = 0

        batch_loss, cat_losses = criterion(pred_list, label_list)
        
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        
        