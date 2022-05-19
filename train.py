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

BATCH_SIZE = 8
NUM_CLASSES = 5
EPOCH = 100

class AverageMeter(object):
  def __init__(self):
      self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

device = torch.device('cuda')

train_dataset = Derma_dataset('/opt/ml/input/data/train', transform=None)
val_dataset = Derma_dataset('/opt/ml/input/data/val', transform=None)

train_dataloader = DataLoader(train_dataset, 
                              batch_size = BATCH_SIZE, 
                              shuffle=True,
                              num_workers=4,
                              drop_last=True)

val_dataloader = DataLoader(val_dataset,
                            batch_size = BATCH_SIZE,
                            num_workers=4,
                            drop_last=True)


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

        train_pred_list = [torch.argmax(pred_list[i], dim=-1) for i in range(len(pred_list))]
        train_oil_acc = (train_pred_list[0] == label_list[0]).sum().item() / BATCH_SIZE
        train_sen_acc = (train_pred_list[1] == label_list[1]).sum().item() / BATCH_SIZE
        train_pig_acc = (train_pred_list[2] == label_list[2]).sum().item() / BATCH_SIZE
        train_wri_acc = (train_pred_list[3] == label_list[3]).sum().item() / BATCH_SIZE
        train_hyd_acc = (train_pred_list[4] == label_list[4]).sum().item() / BATCH_SIZE
        train_total_acc = (train_oil_acc + train_sen_acc + train_pig_acc + train_wri_acc + train_hyd_acc) / 5
        train_total_loss = batch_loss
 
    model.eval()

    val_total_acc = 0
    val_total_loss = 0
    val_oil_acc, val_sen_acc, val_pig_acc, val_wri_acc, val_hyd_acc = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    val_oil_loss, val_sen_loss, val_pig_loss, val_wri_loss, val_hyd_loss = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    for (x, Y) in val_dataloader:
        x = x.to(device)
        
        label_list = [Y[cat].to(device) for cat in Y.keys()]
        with torch.no_grad():
            pred_list = model(X)
        
        batch_loss, cat_losses = criterion(pred_list, label_list)

        pred_label = [torch.argmax(pred_list[i], dim=-1) for i in range(len(pred_list))]
        

        oil_acc = (pred_label[0] == label_list[0]).sum().item() / BATCH_SIZE
        sen_acc = (pred_label[1] == label_list[1]).sum().item() / BATCH_SIZE
        pig_acc = (pred_label[2] == label_list[2]).sum().item() / BATCH_SIZE
        wri_acc = (pred_label[3] == label_list[3]).sum().item() / BATCH_SIZE
        hyd_acc = (pred_label[4] == label_list[4]).sum().item() / BATCH_SIZE
        total_acc = (oil_acc + sen_acc + pig_acc + wri_acc + hyd_acc) / 5
        oil_loss, sen_loss, pig_loss, wri_loss, hyd_loss = cat_losses

        val_oil_acc.update(oil_acc, BATCH_SIZE)
        val_sen_acc.update(sen_acc, BATCH_SIZE)
        val_pig_acc.update(pig_acc, BATCH_SIZE)
        val_wri_acc.update(wri_acc, BATCH_SIZE)
        val_hyd_acc.update(hyd_acc, BATCH_SIZE)

        val_oil_loss.update(oil_loss, BATCH_SIZE)
        val_sen_loss.update(sen_loss, BATCH_SIZE)
        val_pig_loss.update(pig_loss, BATCH_SIZE)
        val_wri_loss.update(wri_loss, BATCH_SIZE)
        val_hyd_loss.update(hyd_loss, BATCH_SIZE)

    val_oil_acc = val_oil_acc.avg
    val_sen_acc = val_sen_acc.avg
    val_pig_acc = val_pig_acc.avg
    val_wri_acc = val_wri_acc.avg
    val_hyd_acc = val_hyd_acc.avg

    val_oil_loss = val_oil_loss.avg
    val_sen_loss = val_sen_loss.avg
    val_pig_loss = val_pig_loss.avg
    val_wri_loss = val_wri_loss.avg
    val_hyd_loss = val_hyd_loss.avg

    val_total_acc = (val_oil_acc + val_sen_acc + val_pig_acc + val_wri_acc + val_hyd_acc) / 5
    val_total_loss = (val_oil_loss + val_sen_loss + val_pig_loss + val_wri_loss + val_hyd_loss) / 5

    print(f"Epoch [{i}/{EPOCH}] | Train Total Loss {train_total_loss:.4f} | Train Total Acc {train_total_acc:.4f} ")
    print(f"Val Total Loss {val_total_loss:.4f} | Val Total Acc {val_total_acc:.4f} ")
    print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
    print(f"| Oil Acc {val_oil_acc:.4f} | Oil Loss {val_oil_loss:.4f} |")
    print(f"| Sen Acc {val_sen_acc:.4f} | Sen Loss {val_sen_loss:.4f} |")
    print(f"| Pig Acc {val_pig_acc:.4f} | Pig Loss {val_pig_loss:.4f} |")
    print(f"| Wri Acc {val_wri_acc:.4f} | Wri Loss {val_wri_loss:.4f} |")
    print(f"| Hyd Acc {val_hyd_acc:.4f} | Hyd Loss {val_hyd_loss:.4f} |")
    print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")