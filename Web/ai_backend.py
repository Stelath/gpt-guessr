import os
import numpy as np
from PIL import Image

import torch
from torchvision import transforms

from gpt_guessr import GPTGuessr, GPTGuessrConfig

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize,
    ])

model_config = GPTGuessrConfig(num_channels=9)
model = GPTGuessr(model_config)
# print(torch.load("guessr_006.pth/pytorch_model.bin", map_location=torch.device('cpu')).keys())
model.load_state_dict(torch.load("guessr_034.pth/pytorch_model.bin", map_location=torch.device('cpu')))
model.eval()

countries = np.load("countries.npy", allow_pickle=True)

def get_coords():
    items = []
    for i in range(0, 3):
        data_path = f'Image{i + 1}.png'
        img = Image.open(data_path)
        items.append(img.convert("RGB"))
    
    for i, item in enumerate(items):
        items[i] = preprocess(item)
    
    items = torch.cat((items[0], items[1], items[2]), dim=0).unsqueeze(0)
    
    pred_country, pred_coords = model(items)
    pred_coords = pred_coords.detach().numpy()[0]
    pred_coords = [str(pred_coords[0]), str(pred_coords[1])]
    
    print(countries[torch.argmax(pred_country)])
    return pred_coords
    
