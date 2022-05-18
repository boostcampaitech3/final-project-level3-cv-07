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
        self.data = self.load_json(data_dir)
        
    
    def load_json(data_dir):
        json_list = glob.glob(data_dir + '/JPEGImages/*.json')
        
        data = []
        
        for json_file in json_list:
            with open(json_file, 'r') as f:
                json_data = json.load(f)
            
            file_name = json_data['file_name']
            del json_data['file_name']
            data.append([data_dir + '/JPEGImages/{}'.format(file_name), json_data])
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index) -> tuple(torch.Tensor, dict):
        y = self.data[index][1]
        
        x = cv2.imread(self.data[index][0])
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
            x = self.transform(image=x)["image"]
        else:
            totensor = transforms.ToTensor()
            x = totensor(x)
            
        
        return x, y