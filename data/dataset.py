
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
    
    def __init__(self, data_dir : str, part = None, cat = None, transform=None) -> None:
        super().__init__()
        
        self.transform = transform
        self.part_table = [['oil', 'sensitive', 'pigmentation'],
                           ['oil', 'sensitive', 'wrinkle'],
                           ['oil', 'sensitive', 'pigmentation', 'wrinkle'],
                           ['sensitive', 'wrinkle', 'hydration'],
                           ['oil', 'sensitive', 'pigmentation', 'wrinkle', 'hydration']]
        self.select_idx = part
        self.select_cat = cat
        
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
            if os.path.exists(path + '/JPEGImages/{}'.format(json_data['file_name'])):
                file_name = json_data['file_name']
            elif os.path.exists(path + '/JPEGImages/{}'.format(json_data['file_name'].split('.')[0] + '.png')):
                file_name = json_data['file_name'].split('.')[0] + '.png'
            else:
                continue
            
            if self.select_idx is not None:
                if self.select_idx != json_data['part']:
                    continue
                else:
                    if self.select_cat is None:
                        for cat in self.part_table[self.select_idx]:
                            if json_data[cat] < 0:
                                continue
                            label_data[cat] = json_data[cat]
                    else:
                        if json_data[self.select_cat] < 0:
                            json_data[self.select_cat] = 5
                        label_data[self.select_cat] = json_data[cat]           
            else:
                if self.select_cat is None:
                    for cat in self.part_table[-1]:
                        if json_data[cat] < 0:
                            continue
                        label_data[cat] = json_data[cat]
                else:
                    if self.select_cat not in self.part_table[json_data['part']]:
                        continue
                    else:
                        if json_data[self.select_cat] < 0:
                            continue
                        label_data[self.select_cat] = json_data[self.select_cat]

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
                resize = A.Resize(384, 512) # (960, 1024)
                x = resize(image=x)['image']
                x = A.normalize(img=x, mean=[0.697, 0.560, 0.482], std=[0.717, 0.582, 0.505])
            elif self.data[index][2] == 1:
                resize = A.Resize(384, 512) # (768, 1024)
                x = resize(image=x)['image']
                x = A.normalize(img=x, mean=[0.567, 0.480, 0.420], std=[0.633, 0.537, 0.474])
            elif self.data[index][2] == 2:
                resize = A.Resize(384, 512) # (512, 1024)
                x = resize(image=x)['image']
                x = A.normalize(img=x, mean=[0.739, 0.595, 0.516], std=[0.756, 0.612, 0.535])
            else:
                resize = A.Resize(384, 512) # (512, 1024)
                x = resize(image=x)['image']
                x = A.normalize(img=x, mean=[0.590, 0.477, 0.413], std=[0.628, 0.512, 0.448])
            totensor = ToTensorV2()
            x = totensor(image=x)['image'].float()
        return x, y