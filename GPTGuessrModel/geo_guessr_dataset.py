import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import os, os.path
from PIL import Image
import pandas as pd

class GeoGuessrDataset(Dataset):
    def __init__(self, df_file, data_dir, transforms=None):
        self.df = pd.read_pickle(os.join(data_dir, df_file))
        self.data_dir = data_dir

    def __len__(self):
        return len(os.listdir(self.data_dir)) - 1

    def __getitem__(self, idx):
        items = []
        df_item = self.df.iloc[idx]
        file_path = df_item['file_path']
        country = df_item['country']
        coords = df_item['coords']
        for i in range(0, 3):
            data_path = os.path.join(self.data_dir, f'{file_path}_{i}.jpg')
            items.append(Image.open(data_path))
        
        if self.transforms:
            for i, item in enumerate(items):
                items[i] = self.transforms(item)
        
        items = torch.stack(items)
        
        item = {'image': items, 'country': F.one_hot(torch.Tensor(country).long(), num_classes=177), 'coords': torch.Tensor(coords, dtype=torch.float32)}
        
        return item
    