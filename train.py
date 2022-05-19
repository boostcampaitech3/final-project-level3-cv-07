from re import U

from cv2 import CALIB_THIN_PRISM_MODEL
from data.dataset import Derma_dataset
from model.model import Convnext_custom
from model.losses import FocalLoss, Derma_FocalLoss
from model.metric import ArcMarginProduct
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import random

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Classfier')
    parser.add_argument('--epoch', type=int, default=50, help='training epoch setting')
    parser.add_argument('--val-interval', type=int, default=1, help='validation interval')
    parser.add_argument('--model size', type=str, default='tiny', help='model size config, ex) tiny, small, base, large, xlarge')
    parser.add_argument('--save-path', help='the dir to save model')
    parser.add_argument('--max_ckpt', type=int, default=3, help='maximum keep ckpt files in save_dir')
    parser.add_argument('--load-from', help='the checkpoint file to load weights from')
    parser.add_argument('--no-validate', help='whether not to evaluate the validation set during training')
    parser.add_argument('--seed', type=int, default=2022, help='random Seed setting')
    


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

BATCH_SIZE = 8
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


model = Convnext_custom('tiny')

criterion = Derma_FocalLoss(gamma=2).to(device)

# metric_fc = ArcMarginProduct(model.get_last_dim(), NUM_CLASSES)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(),
                              lr=0.001, weight_decay=0.05)

cat_list = ['oil', 'sensitive', 'pigmentation', 'wrinkle', 'hydration']

for i in range(EPOCH):
    model.train()
    pbar = tqdm(enumerate(train_dataloader), unit='batch')
    epoch_loss = {cat : 0 for cat in cat_list}
    epoch_acc = {cat : 0 for cat in cat_list}
    
    for idx, data in pbar:
        X, Ys = data
        X = X.to(device)
        
        
        pred_dict = model(X)
        acc_list = []
        
        for cat in cat_list:
            pred = torch.argmax(pred_dict[cat], dim=-1)
            gt_y = Ys[cat].to(device)
            
            pred = pred[gt_y != 5]
            gt_y = gt_y[gt_y != 5]
            
            corr = torch.sum(pred.data == gt_y.data) / pred.shape[0]
            acc_list.append(corr)
            
        batch_loss, cat_losses = criterion(pred_dict, Ys)
        
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        for idx, cat in enumerate(cat_list):
            epoch_loss[cat] += cat_losses[idx].item()
            epoch_acc[cat] += acc_list[idx].item()
        
        pbar.set_postfix(total_loss=batch_loss.item(), 
                         oil_loss=cat_losses[0].item(), sen_loss=cat_losses[1].item(),
                         pig_loss=cat_losses[2].item(), wri_loss=cat_losses[3].item(), hyd_loss=cat_losses[4].item())
        print('\nOli_Acc : {:0.4f} | Sen_Acc : {:0.4f}\nPig_Acc : {:0.4f} | Wri_Acc : {:0.4f} | Hyd_Acc : {:0.4f}'.format(acc_list[0].item(), acc_list[1].item(), acc_list[2].item(), acc_list[3].item(), acc_list[4].item()))

    print('=' * 25 + ' Epoch End! ' + '=' * 25)
    print('Total Loss : {:0.4f} | Loss_Oil : {:0.4f} | Loss_Sen : {:0.4f} | Loss_Pig : {:0.4f} | Loss_Wri : {:0.4f} | Loss_Hyd : {:0.4f}'.format(
        sum(epoch_loss.values()) / len(train_dataloader), epoch_loss['oil'] / len(train_dataloader), epoch_loss['sensitive'] / len(train_dataloader),
        epoch_loss['pigmentation'] / len(train_dataloader), epoch_loss['wrinkle'] / len(train_dataloader), epoch_loss['hydration'] / len(train_dataloader)
    ))
    print('Total Acc  : {:0.4f} | Acc_Oil  : {:0.4f} | Acc_Sen  : {:0.4f} | Acc_Pig  : {:0.4f} | Acc_Wri  : {:0.4f} | Acc_Hyd  : {:0.4f}'.format(
        sum(epoch_acc.values()) / len(train_dataloader), epoch_acc['oil'] / len(train_dataloader), epoch_acc['sensitive'] / len(train_dataloader),
        epoch_acc['pigmentation'] / len(train_dataloader), epoch_acc['wrinkle'] / len(train_dataloader), epoch_acc['hydration'] / len(train_dataloader)
    ))
    print('=' * 70)