import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import os, os.path
from PIL import Image
import pandas as pd

class GeoGuessrDataset(Dataset):
    def __init__(self, df_file, data_dir, size=244, transform=None):
        self.df = pd.read_pickle(os.path.join(data_dir, df_file))
        self.data_dir = data_dir
        self.df.dropna(inplace=True)
        self.df = self.df.reset_index()
        self.transform = transform
        self.size = size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        items = []
        df_item = self.df.iloc[idx]
        file_path = df_item['file_name']
        country = df_item['country']
        coords = df_item['coords']
        for i in range(0, 3):
            data_path = os.path.join(self.data_dir, f'{file_path}_{i}.jpg')
            try:
                img = Image.open(data_path)
            except:
                img = Image.new('RGB', (self.size, self.size))
            items.append(img.convert("RGB"))
        
        if self.transform:
            for i, item in enumerate(items):
                items[i] = self.transform(item)
        
        items = torch.cat((items[0], items[1], items[2]), dim=0)
        item = {'images': items, 'country': F.one_hot(torch.tensor(country).long(), num_classes=177).type(torch.float32), 'coords': torch.tensor(coords, dtype=torch.float32)}
        
        return item
    