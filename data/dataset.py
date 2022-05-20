from sre_constants import CATEGORY_LINEBREAK
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from albumentations.pytorch import ToTensorV2
from PIL import Image
import torch
import cv2
import os, glob
import json
import numpy as np
import albumentations as A



class Derma_dataset(Dataset):
    
    def __init__(self, data_dir : str, select_idx : int, transform=None) -> None:
        super().__init__()
        if select_idx is not None:
            if select_idx < 0 or select_idx > 3:
                raise ValueError('select part idx are 0 ~ 3, you select {select_idx}')

        
        self.transform = transform
        self.part_table = [['oil', 'sensitive', 'pigmentation'],
                           ['oil', 'sensitive', 'wrinkle'],
                           ['oil', 'sensitive', 'pigmentation', 'wrinkle'],
                           ['sensitive', 'wrinkle', 'hydration']]
        self.select_idx = select_idx
        
        self.data = self.load_json(data_dir)
        
    def load_json(self, path):
        print('load json files from {}'.format(path))
        json_list = glob.glob(path + '/JPEGImages/*.json')
        print('total json files : {}'.format(len(json_list)))
    
        data = []    
        for json_file in json_list:
            with open(json_file, 'r') as f:
                json_data = json.load(f)
            
            label_data = {}
            if json_data['part'] != self.select_idx:
                continue

            file_name = json_data['file_name']
            
            for cat in self.part_table[self.select_idx]:
                if json_data[cat] < 0:
                    json_data[cat] = 5
                label_data[cat] = json_data[cat]
            
            data.append([path + '/JPEGImages/{}'.format(file_name), label_data, json_data['part']])

                
        
        print('Data Load Success, Total Data length : {}'.format(len(data)))
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        y = self.data[index][1]
        
        
        x = cv2.imread(self.data[index][0])
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
            x = self.transform(image=x)["image"]
        else:
            if self.data[index][2] == 0:
                resize = A.Resize(960, 1024)
                x = resize(image=x)['image']
            elif self.data[index][2] == 1:
                resize = A.Resize(768, 1024)
                x = resize(image=x)['image']
            elif self.data[index][2] == 2:
                resize = A.Resize(512, 1024)
                x = resize(image=x)['image']
            else:
                resize = A.Resize(512, 1024)
                x = resize(image=x)['image']
            totensor = ToTensorV2()
            x = totensor(image=x)['image'].float()
        return x, y