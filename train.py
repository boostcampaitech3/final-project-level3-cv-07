import os, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import random
from tqdm import tqdm
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from data.dataset import Derma_dataset
from model.model import Convnext_custom, Resnet50, convnext_large, convnext_tiny, convnext_small, convnext_base
from model.losses import FocalLoss, Derma_FocalLoss, Derma_CELoss, QuadraticKappaLoss, F1_Loss
from model.metric import ArcMarginProduct
import wandb
from utils import create_matrix, print_report

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Classfier')
    parser.add_argument('--epoch', type=int, default=25, help='training epoch setting')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size setting')
    parser.add_argument('--cat', type=str, default='pigmentation', help='select category to train')
    parser.add_argument('--val-interval', type=int, default=1, help='validation interval')
    parser.add_argument('--log-interval', type=int, default=10, help='training log interval')
    parser.add_argument('--model-size', type=str, choices=['tiny', 'small', 'base', 'large', 'xlarge'], default='tiny', 
                        help='model size config, ex) tiny, small, base, large, xlarge')
    parser.add_argument('--save-path', default= '../work_dir/best', help='the dir to save model')
    parser.add_argument('--save-interval', type=int, default=5, help='save pth interval, based epoch')
    parser.add_argument('--max_ckpt', type=int, default=2, help='maximum keep ckpt files in save_dir')
    parser.add_argument('--load-from', help='the checkpoint file to load weights from')
    parser.add_argument('--no-validate', action='store_false', help='whether not to evaluate the validation set during training')
    parser.add_argument('--seed', type=int, default=2022, help='random Seed setting')
    parser.add_argument('--lw_qwk', type=float, default=1.0, help='loss weight setting for qwk')
    parser.add_argument('--lw_focal', type=float, default=1.0, help='loss weight setting for focal')
    parser.add_argument('--lw_f1', type=float, default=1.0, help='loss weight setting for f1')
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate setting')
    
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
    save_name = os.path.join(save_path, type + '_' + f'{epoch_cnt:.3f}' + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name

def check_pth(save_path, max_ckpt):
    pth_list = glob.glob(save_path + '/*.pth')
    pth_list = sorted(pth_list, key=lambda x : float(x.split('/')[-1].split('_')[-1][:5]))
    
    while len(pth_list) >= max_ckpt:
        if os.path.exists(pth_list[0]):
            os.remove(pth_list.pop(0))

def change_ordinal(labels, num_classes):
    ord_labels = (2 * labels + 1) / (2 * num_classes)
    return ord_labels

def change_class(ord_labels, num_classes, device='cuda'):
    delta = 1.0 / num_classes
    labels = torch.zeros(ord_labels.size()).to(device)
    for i in range(num_classes):
        labels += torch.where((ord_labels >= delta * i) & (ord_labels < delta * (i+1)), i, 0)
    labels += torch.where(ord_labels == 1, num_classes-1, 0)
    return labels

def wandb_vz_img(img_tensor, label_list, pred_list, cat=None, part=None):
    caption = 'GT->PRED\n'
    
    caption += '{}:{}->{}\t'.format(cat, label_list[0], pred_list[0])

    img = wandb.Image(img_tensor[0], caption=caption)
            
    return img


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
        
cat_weight = {'oil': torch.tensor([128.0250,   3.7525,   1.8125,   6.5825,  91.0800]),
            'sensitive': torch.tensor([  2.6020,   2.2460,   8.5800,  90.7700, 260.1100]),
            'pigmentation': torch.tensor([ 4.6800,  2.2733,  3.9233, 14.2667, 45.4533]),
            'wrinkle': torch.tensor([  3.7500,   2.6325,   5.8975,  42.8225, 275.3750]),
            'hydration': torch.tensor([16.9700,  3.7300,  2.4300,  5.1000, 15.0800])}

# cat_num_classes = {'oil' : 3,
#                    'sensitive' : 3,
#                    'pigmentation' : 4,
#                    'wrinkle' : 4,
#                    'hydration' : 5}

# cat_data_nums = {'cheek' : {'oil' : torch.tensor([6, 124, 92, 10, 0]),
#                              'sensitive' : torch.tensor([80, 110, 40, 2, 0]),
#                              'pigmentation' : torch.tensor([24, 132, 46, 22, 8])},
#                  'upper_face' : {'oil' : torch.tensor([0, 37, 64, 14, 0]),
#                                  'sensitive' : torch.tensor([39, 52, 24, 0, 0]),
#                                  'wrinkle' : torch.tensor([62, 42, 3, 8, 0])},
#                  'mid_face' : {'oil' : torch.tensor([2, 49, 59, 5, 0]),
#                                'sensitive' : torch.tensor([50, 57, 8, 0, 0]),
#                                'pigmentation' : torch.tensor([10, 67, 22, 4, 12]),
#                                'wrinkle' : torch.tensor([12, 37, 58, 4, 4])},
#                  'lower_face' : {'sensitive' : torch.tensor([65, 44, 4, 2, 0]),
#                                  'wrinkle' : torch.tensor([102, 5, 4, 4, 0])}
#                 }

# cat_weight = {'oil': torch.tensor([ 2.1193,  2.1488, 15.9310]),
#                'sensitive': torch.tensor([2.4658, 2.1939, 7.2125]),
#                'pigmentation': torch.tensor([10.2059,  1.7437,  5.1029,  7.5435]),
#                'wrinkle': torch.tensor([ 1.9602,  4.1071,  5.3077, 17.2500])}

# cat_weight_drop_cheek = {'sensitive': torch.tensor([2.3763, 2.2163, 7.8136]),
#                           'pigmentation': torch.tensor([10.5000,  1.7368,  5.1333,  7.4516]),
#                           'oil': torch.tensor([ 2.6136,  1.8699, 12.1053]),
#                           'wrinkle': torch.tensor([ 1.9828,  4.5635,  4.5276, 17.9688])}

def main():
    arg = parse_args()
    set_seed(arg.seed)
    BATCH_SIZE = arg.batch_size
    EPOCH = arg.epoch
    # PART = 2
    cat = arg.cat
    NUM_CLASSES = 5
    device = torch.device('cuda')

    transform = A.Compose([
                # A.LongestMaxSize(),
                # A.PadIfNeeded(border_mode=cv2.BORDER_CONSTANT, value=0),
                A.Resize(384, 512),
                A.Rotate(limit=5,
                        border_mode= cv2.BORDER_CONSTANT, 
                        value=0,
                        p=0.5
                        ),
                A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4),
                A.RGBShift(p=0.5),
                # A.CLAHE(clip_limit=(2, 2), always_apply=True),
                A.HorizontalFlip(),
                A.Normalize(
                    mean=(0.65490196, 0.53333333, 0.45882353),
                    std=(0.18431373, 0.16078431, 0.14901961)),
                ToTensorV2()
            ])
    
    val_trasform = A.Compose([
                            # A.LongestMaxSize(),
                            # A.PadIfNeeded(border_mode=cv2.BORDER_CONSTANT, value=0),
                            A.Resize(384, 512),
                            # A.CLAHE(clip_limit=(2, 2), always_apply=True),
                            A.Normalize(mean=(0.65490196, 0.53333333,0.45882353),
                                        std=(0.18431373, 0.16078431, 0.14901961)),
                            ToTensorV2()
                        ])
    
    train_dataset = Derma_dataset(['/opt/ml/input/data/train_nonbg_final', '/opt/ml/input/data/train'], cat=cat, transform=transform)
    val_dataset = Derma_dataset('/opt/ml/input/data/val', cat=cat, transform=val_trasform)

    train_dataloader = DataLoader(train_dataset, 
                                batch_size = BATCH_SIZE, 
                                shuffle=True,
                                num_workers=4,
                                drop_last=True)


    val_dataloader = DataLoader(val_dataset,
                                batch_size=BATCH_SIZE,
                                num_workers=4,
                                drop_last=True)


    model = convnext_tiny(num_classes=21841, pretrained=True, in_22k=True)
    model.head = nn.Linear(model.get_last_dim(), NUM_CLASSES)
    model._init_weights(model.head)
    
    # reg_parts = nn.Sequential(nn.Linear(model.get_last_dim(), 1), 
    #                            nn.Sigmoid()).to(device)
    # cat_parts = nn.Linear(model.get_last_dim(), 5).to(device)
    
    if arg.load_from is not None:
        model.load_state_dict(torch.load(arg.load_from))

    # reg_criterion = nn.SmoothL1Loss()
    cat_criterion = QuadraticKappaLoss(num_classes=NUM_CLASSES).to(device)
    cat_criterion_ce = FocalLoss(weight=cat_weight[cat].to(device), ignore_index=5)
    cat_criterion_f1 = F1_Loss(NUM_CLASSES)
    # metric_fc = ArcMarginProduct(model.get_last_dim(), 5).to(device)
    model.to(device)

    # optimizer = torch.optim.AdamW([{'params' : model.parameters()}, {'params' : metric_fc.parameters()}], lr=0.0001, weight_decay=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arg.lr, weight_decay=0.05)

    # scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.99)

    best_valid = 0
    
    wandb.init(project='find_sen', entity='final-project', name=f'best_{arg.cat[:3]}/QWK{arg.lw_qwk:.4f}+Focal{arg.lw_focal:.4f}+f1{arg.lw_f1:.4f}/lr{arg.lr}')
    wandb.config.update(arg)
    wandb.watch(model)
    
    for epoch in range(EPOCH):
        wandb.log({'epoch' : epoch})
        model.train()
        
        train_arc_acc = 0
        train_cat_acc = 0
        # train_reg_acc = 0
        train_arc_loss = 0
        train_cat_loss = 0
        # train_reg_loss = 0
        # train_total_acc = 0
        # train_total_loss = 0
        for idx, data in tqdm(enumerate(train_dataloader), unit='Iter'):
            X, Ys = data
            X = X.to(device)
            Y = Ys[cat].to(device)
            
            
            # feat = model.forward_features(X)
            
            # reg_preds = reg_parts(feat)
            # reg_preds = reg_preds.view((-1,))

            # reg_cvt_preds = change_class(reg_preds, NUM_CLASSES).to(torch.long)
            
            # reg_Y = torch.where(Y == 5, reg_cvt_preds, Y)
            # ord_Y = change_ordinal(Y, NUM_CLASSES)
            
            # feat = model.forward_features(X)
            # arc_preds = metric_fc(feat, Y)

            # arc_loss_ce = cat_criterion_ce(arc_preds, Y)
            # arc_preds = F.softmax(arc_preds, dim=-1)

            # arc_loss_qwk = cat_criterion(arc_preds, Y)
            
            
            # arc_loss = arc_loss_qwk + (arc_loss_ce * 0.3)
            
            # optimizer.zero_grad()
            # arc_loss.backward()
            # optimizer.step()

            # for param in model.parameters():
                # param.requires_grad = False
            # for param in model.head.parameters():
                # param.requires_grad = True
            
            cat_preds = model(X)
            cat_Y = torch.where(Y == 5, torch.argmax(cat_preds, dim=-1), Y)
            cat_loss_ce = cat_criterion_ce(cat_preds, cat_Y)
            cat_loss_f1 = cat_criterion_f1(cat_preds, cat_Y)
            cat_preds = F.softmax(cat_preds, dim=-1)
            # reg_loss = reg_criterion(reg_preds, ord_Y)
            cat_loss_qwk = cat_criterion(cat_preds, cat_Y)
            
            optimizer.zero_grad()
            cat_loss = (cat_loss_qwk * arg.lw_qwk) + (cat_loss_ce * arg.lw_focal) + (cat_loss_f1 * arg.lw_f1)
            cat_loss.backward()
            optimizer.step()
            
            # for param in model.parameters():
                # param.requires_grad = True
            
            # reg_acc = (reg_cvt_preds == reg_Y).sum().item() / BATCH_SIZE
            # arc_acc = (torch.argmax(arc_preds, dim=-1) == Y).sum().item() / BATCH_SIZE
            cat_acc = (torch.argmax(cat_preds, dim=-1) == cat_Y).sum().item() / BATCH_SIZE
            
            
            # train_reg_acc += reg_acc
            # train_arc_acc += arc_acc
            train_cat_acc += cat_acc
            # train_total_acc += reg_acc
            # train_arc_loss += arc_loss.item()
            train_cat_loss += cat_loss.item()
            # train_reg_loss += reg_loss.item()
            # train_total_loss += loss.item()

            # train accuracy와 loss에서는 그냥 50iter 마다 그때의 acc, loss 출력
            if ((idx+1) % arg.log_interval) == 0:
                print("  Iter[{} / {}] \n \
                        | Train_Cat_Acc  : {:.4f} \n \
                        | Train_Cat_loss : {:.4f}".format(
                    idx + 1, len(train_dataloader), 
                    train_cat_acc / arg.log_interval,
                    train_cat_loss / arg.log_interval,
                ))
                wandb.log({
                        # 'train_arc_acc' : train_arc_acc / arg.log_interval,
                        # 'train_arc_loss' : train_arc_loss / arg.log_interval,
                        'train_cat_acc' : train_cat_acc / arg.log_interval,
                        'train_cat_loss' : train_cat_loss / arg.log_interval,
                })
                # train_arc_acc = 0
                train_cat_acc = 0
                # train_reg_acc = 0
                # train_arc_loss = 0
                train_cat_loss = 0
                # train_reg_loss = 0
                # train_total_acc = 0
                # train_total_loss = 0

        # scheduler.step()

        if (arg.save_path is not None) & ((epoch + 1) % arg.save_interval == 0):
            save_model(model, arg.save_path + f'_{arg.cat}', epoch+1, arg.max_ckpt)
            
        model.eval()

        y_preds = []
        y_trues = []
                
        # val_reg_acc_meter = AverageMeter()
        val_cat_acc = []
        # val_reg_loss_meter = AverageMeter()
        val_cat_loss = []
        # val_total_acc_meter = AverageMeter()
        # val_total_loss_meter = AverageMeter()
        
        vz_img = []
        if arg.no_validate & ((epoch + 1) % arg.val_interval == 0):
            for (val_X, val_Y) in val_dataloader:
                val_X = val_X.to(device)
                val_Y = val_Y[cat].to(device)
                
                
                with torch.no_grad():
                    val_cat_preds = model(val_X)
                
                # val_reg_preds = reg_parts(val_feat)
                # val_reg_preds = val_reg_preds.view((-1,))
                
                # val_reg_cvt_preds = change_class(val_reg_preds, NUM_CLASSES).to(torch.long)
                # val_reg_Y = torch.where(val_Y == 5, val_reg_cvt_preds, val_Y)
                # val_ord_Y = change_ordinal(val_reg_Y, NUM_CLASSES)
                
                # val_cat_preds = cat_parts(val_feat)
                val_cat_Y = torch.where(val_Y == 5, torch.argmax(val_cat_preds, dim=-1), val_Y)
                val_cat_loss_ce = cat_criterion_ce(val_cat_preds, val_cat_Y)
                val_cat_loss_f1 = cat_criterion_f1(val_cat_preds, val_cat_Y)
                val_cat_preds = F.softmax(val_cat_preds, dim=-1)
                
                # val_reg_loss = reg_criterion(val_reg_preds, val_ord_Y)
                val_cat_loss_qwk = cat_criterion(val_cat_preds, val_cat_Y)
                val_cat_preds = torch.argmax(val_cat_preds, dim=-1)
                
                val_cat_loss.append(((val_cat_loss_qwk.item() * arg.lw_qwk) + (val_cat_loss_ce.item() * arg.lw_focal) + (val_cat_loss_f1.item() * arg.lw_f1)) / BATCH_SIZE)
                
                # val_reg_acc = (val_reg_cvt_preds == val_reg_Y).sum().item() / BATCH_SIZE
                val_cat_acc.append(((val_cat_preds == val_cat_Y).sum().item()) / BATCH_SIZE)
                
                # val_reg_acc_meter.update(val_reg_acc, BATCH_SIZE)
                # val_reg_loss_meter.update(val_reg_loss, BATCH_SIZE)
                
                if len(vz_img) <= 106 or vz_img:
                    vz_img.append(wandb_vz_img(val_X, val_Y, val_cat_preds, cat=cat))
                
                # val_total_acc_meter.update(val_acc, 2)
                # val_total_loss_meter.update(val_loss, 2)

                # 배치 단위 pred / label들을 y_preds, y_trues list에 저장.
                y_pred = val_cat_preds.data.cpu().numpy() # model_output.data.cpu().numpy()
                y_true = val_cat_Y.data.cpu().numpy()         # dataset_Y.data.cpu().numpy()
                
                y_preds.extend(y_pred)
                y_trues.extend(y_true)
                

            print(f"Epoch [{epoch+1}/{EPOCH}] \n \
                    Val Cat Loss : {sum(val_cat_loss) / len(val_cat_loss):.4f} \n \
                    Val Cat Acc  : {sum(val_cat_acc) / len(val_cat_acc):.4f}")
            report = print_report(y_trues, y_preds, NUM_CLASSES)

            if arg.save_path is not None:
                create_matrix(y_trues, y_preds, arg.save_path + f'_{arg.cat}', epoch+1, NUM_CLASSES)
                if report['kappa'] + report['macro_f1_score'] > best_valid:
                    best_valid = report['kappa'] + report['macro_f1_score']
                    save_model(model, os.path.join(arg.save_path + f'_{arg.cat}', "best"), best_valid, 2, f"best_qwk_{report['kappa']:.3f}_f1_{report['macro_f1_score']:.3f}")
                    print("*** Save the best model ***\n")
            wandb.log(report)
            wandb.log({
                'kappa+f1' : report['kappa'] + report['macro_f1_score'],
                'val_cat_acc' : sum(val_cat_acc) / len(val_cat_acc),
                'val_cat_loss' : sum(val_cat_loss) / len(val_cat_loss),
                'batch[0]' : vz_img
            })

if __name__ == '__main__':
    main()