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

    def __init__(self, data_dir, transform=None) -> None:
        super().__init__()
        
        self.transform = transform
        self.cat_list = ['oil', 'sensitive', 'pigmentation', 'wrinkle', 'hydration']
        self.data = self.load_json(data_dir)
    
    def load_json(self, path):
        print('load json files from {}'.format(path))
        json_list = glob.glob(path + '/JPEGImages/*.json')
        print('total json files : {}'.format(len(json_list)))
    
        data = []    
        for json_file in json_list:
            with open(json_file, 'r') as f:
                json_data = json.load(f)
            
            file_name = json_data['file_name']
            
            for cat in self.cat_list:
                if json_data[cat] < 0:
                    json_data[cat] = 5
                
            del json_data['file_name']
            del json_data['part']
            data.append([path + '/JPEGImages/{}'.format(file_name), json_data])
        
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
            resizer = A.Resize(1024, 1024)
            totensor = ToTensorV2()
            x = resizer(image=x)['image']
            x = totensor(image=x)['image']
            x = x.float()

        return x, y