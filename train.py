from codecs import ignore_errors
import os, glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import random
from tqdm import tqdm
import argparse

from data.dataset import Derma_dataset
from model.model import Convnext_custom
from model.losses import FocalLoss, Derma_FocalLoss, Derma_CELoss
from model.metric import ArcMarginProduct


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Classfier')
    parser.add_argument('--epoch', type=int, default=50, help='training epoch setting')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size setting')
    parser.add_argument('--val-interval', type=int, default=1, help='validation interval')
    parser.add_argument('--log-interval', type=int, default=50, help='training log interval')
    parser.add_argument('--model-size', type=str, choices=['tiny', 'small', 'base', 'large', 'xlarge'], default='tiny', 
                        help='model size config, ex) tiny, small, base, large, xlarge')
    parser.add_argument('--save-path', help='the dir to save model')
    parser.add_argument('--save-interval', type=int, default=5, help='save pth interval, based epoch')
    parser.add_argument('--max_ckpt', type=int, default=3, help='maximum keep ckpt files in save_dir')
    parser.add_argument('--load-from', help='the checkpoint file to load weights from')
    parser.add_argument('--no-validate', action='store_false', help='whether not to evaluate the validation set during training')
    parser.add_argument('--seed', type=int, default=2022, help='random Seed setting')
    
    args = parser.parse_args()
    return args

def set_seed(seed : int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
  
def save_model(model, save_path, epoch_cnt, max_ckpt=None, type="epoch"):
    if max_ckpt is not None:
        check_pth(save_path, max_ckpt)
    os.makedirs(save_path, exist_ok=True)
    save_name = os.path.join(save_path, type + '_' + str(epoch_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name

def check_pth(save_path, max_ckpt):
    pth_list = glob.glob(save_path + '*.pth')
    pth_list = sorted(pth_list, key=lambda x : int(x.split('/')[-1].split('.')[0].split('_')[-1]))
    
    while len(pth_list) >= max_ckpt:
        if os.path.exists(pth_list[0]):
            os.remove(pth_list.pop(0))


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
        

def main():
    arg = parse_args()
    set_seed(arg.seed)
    BATCH_SIZE = arg.batch_size
    NUM_CLASSES = 5
    EPOCH = arg.epoch
    PART = 0

    device = torch.device('cuda')

    train_dataset = Derma_dataset('/opt/ml/input/data/train', select_idx=PART, transform=None)
    val_dataset = Derma_dataset('/opt/ml/input/data/val', select_idx=PART, transform=None)

    train_dataloader = DataLoader(train_dataset, 
                                batch_size = BATCH_SIZE, 
                                shuffle=True,
                                num_workers=4,
                                drop_last=True)

    val_dataloader = DataLoader(val_dataset,
                                batch_size = BATCH_SIZE,
                                num_workers=4,
                                drop_last=True)


    model = Convnext_custom(arg.model_size, part=PART)
    
    if arg.load_from is not None:
        model.load_state_dict(torch.load(arg.load_from))

    criterion = Derma_FocalLoss(ignore_index=5, part=PART)

    # metric_fc = ArcMarginProduct(model.get_last_dim(), NUM_CLASSES)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                lr=0.0001, weight_decay=0.05)

    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.5)

    best_accuracy = 0

    for epoch in range(EPOCH):
        model.train()
        
        train_total_acc = 0
        train_total_loss = 0
        for idx, data in tqdm(enumerate(train_dataloader), unit='Iter'):
            X, Ys = data
            X = X.to(device)
            
            label_list = {cat: Ys[cat].to(device) for cat in Ys.keys()}
            pred_list = model(X)
            
            batch_loss = 0

            batch_loss, cat_losses = criterion(pred_list, label_list)
            
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            train_pred_list = {cat: torch.argmax(pred_list[cat], dim=-1) for cat in Ys.keys()}

            # print("\n",train_pred_list)
            # print(label_list)
            
            acc_list = []
            for cat in Ys.keys():
                exept_cnt = (label_list[cat]==5).sum().item()
                if exept_cnt == BATCH_SIZE:
                    continue
                acc = (train_pred_list[cat] == label_list[cat]).sum().item() / (BATCH_SIZE - exept_cnt)
                acc_list.append(acc)

            train_total_acc += np.mean(acc_list)
            train_total_loss += batch_loss.item()

            # train accuracy와 loss에서는 그냥 50iter 마다 그때의 acc, loss 출력
            if ((idx+1) % arg.log_interval) == 0:
                print("  Iter[{} / {}] | Train_Accuracy: {:.4f} | Train_Loss: {:.4f}".format(
                    idx + 1, len(train_dataloader), train_total_acc / arg.log_interval, train_total_loss / arg.log_interval
                ))
                train_total_acc = 0
                train_total_loss = 0

        scheduler.step()

        if (arg.save_path is not None) & ((epoch + 1) % arg.save_interval == 0):
            save_model(model, arg.save_path, epoch+1, arg.max_ckpt)
            
        model.eval()

        val_total_acc = 0
        val_total_loss = 0
        val_oil_acc, val_sen_acc, val_pig_acc, val_wri_acc, val_hyd_acc = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        val_oil_loss, val_sen_loss, val_pig_loss, val_wri_loss, val_hyd_loss = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

        val_acc_list = [val_oil_acc, val_sen_acc, val_pig_acc, val_wri_acc, val_hyd_acc]
        val_loss_list = [val_oil_loss, val_sen_loss, val_pig_loss, val_wri_loss, val_hyd_loss]
        
        val_acc_dict = {cat : acc for cat, acc in zip(['oil', 'sensitive', 'pigmentation', 'wrinkle', 'hydration'], val_acc_list)}
        val_loss_dict = {cat : l for cat, l in zip(['oil', 'sensitive', 'pigmentation', 'wrinkle', 'hydration'], val_loss_list)}
        
        
        if arg.no_validate & ((epoch + 1) % arg.val_interval == 0):
            for (x, Ys) in val_dataloader:
                x = x.to(device)
                
                label_list = {cat: Ys[cat].to(device) for cat in Ys.keys()}
                with torch.no_grad():
                    pred_list = model(x)
                
                batch_loss, cat_losses = criterion(pred_list, label_list)
                
                pred_list = {cat: torch.argmax(pred_list[cat], dim=-1) for cat in Ys.keys()}
                


                acc_list = []
                for i, cat in enumerate(Ys.keys()):
                    exept_cnt = (label_list[cat]==5).sum().item()
                    if exept_cnt == BATCH_SIZE:
                        continue
                    acc = (pred_list[cat] == label_list[cat]).sum().item() / (BATCH_SIZE - exept_cnt)
                    val_acc_dict[cat].update(acc, BATCH_SIZE - exept_cnt)
                    val_loss_dict[cat].update(cat_losses[cat], BATCH_SIZE - exept_cnt)

            val_oil_acc, val_sen_acc, val_pig_acc, val_wri_acc, val_hyd_acc = val_oil_acc.avg, val_sen_acc.avg, val_pig_acc.avg, val_wri_acc.avg, val_hyd_acc.avg
            val_oil_loss, val_sen_loss, val_pig_loss, val_wri_loss, val_hyd_loss = val_oil_loss.avg, val_sen_loss.avg, val_pig_loss.avg, val_wri_loss.avg, val_hyd_loss.avg
            
            val_acc = [acc[cat].avg for cat in Ys.keys()]
            val_total_acc = sum(val_acc) / len(val_acc)
            val_total_loss = val_oil_loss + val_sen_loss + val_pig_loss + val_wri_loss + val_hyd_loss

            print(f"\nEpoch [{epoch+1}/{EPOCH}]  Val Total Loss {val_total_loss:.4f} | Val Total Acc {val_total_acc:.4f}")
            print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
            print(f"| Oil Acc {val_oil_acc:.4f} | Oil Loss {val_oil_loss:.4f} |")
            print(f"| Sen Acc {val_sen_acc:.4f} | Sen Loss {val_sen_loss:.4f} |")
            print(f"| Pig Acc {val_pig_acc:.4f} | Pig Loss {val_pig_loss:.4f} |")
            print(f"| Wri Acc {val_wri_acc:.4f} | Wri Loss {val_wri_loss:.4f} |")
            print(f"| Hyd Acc {val_hyd_acc:.4f} | Hyd Loss {val_hyd_loss:.4f} |")
            print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ\n")

            if val_total_acc > best_accuracy:
                best_accuracy = val_total_acc
                save_model(model, os.path.join(arg.save_path, "best"), epoch+1, 1, "best")
                print("*** Save the best model ***\n")

if __name__ == '__main__':
    main()