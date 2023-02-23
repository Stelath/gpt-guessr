import os
import time
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from transformers import ViTImageProcessor
from geo_guessr_dataset import GeoGuessrDataset
from gpt_guessr import GPTGuessrViT, GPTGuessrConfig, GPTGuessrConvNeXt

from accelerate import Accelerator

from dataclasses import dataclass

@dataclass
class TrainingConfig:
    rgb = True
    image_size = 512  # the generated image resolution
    train_batch_size = 32
    eval_batch_size = 32  # how many images to sample during evaluation
    num_dataloader_workers = 12  # how many subprocesses to use for data loading
    epochs = 60
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    eval_epochs = 1
    save_model_epochs = 2
    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = '/scratch1/korte/GPTGuessr'
    
    data_dir = 'data/'
    df_file = 'dataset.df'

    overwrite_output_dir = True
    seed = 0

def haversine_distance(coord1, coord2):
    R = 6371.0 # Earth's radius in kilometers
    # coord1, coord2 = coord1.clone().detach(), coord2.clone().detach()
    lat1, lon1 = coord1[..., 0], coord1[..., 1]
    lat2, lon2 = coord2[..., 0], coord2[..., 1]
    phi1, phi2 = torch.deg2rad(lat1), torch.deg2rad(lat2)
    delta_phi = torch.deg2rad(lat2 - lat1)
    delta_lambda = torch.deg2rad(lon2 - lon1)
    
    a = torch.sin(delta_phi / 2.0)**2 + \
        torch.cos(phi1) * torch.cos(phi2) * \
        torch.sin(delta_lambda / 2.0)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    distance = R * c
    return distance


def coord_loss_function(p, t):
    dist = haversine_distance(p, t)
    return F.smooth_l1_loss(dist, torch.zeros_like(dist))


def train():    
    config = TrainingConfig()
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            normalize,
        ])

    
    dataset = GeoGuessrDataset(df_file=config.df_file, data_dir=config.data_dir, size=config.image_size, transform=preprocess)

    train_size = int(len(dataset) * 0.95)
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size], generator=torch.Generator().manual_seed(42))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_dataloader_workers)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=config.eval_batch_size, shuffle=True, num_workers=config.num_dataloader_workers)
    
    print(f"Loaded Dataloaders")
    print(f"Training on {len(train_dataset)} locations, evaluating on {len(eval_dataset)} locations")
    
    ### TRAIN GPTGuessr ###
    # gpt_guessr_config = GPTGuessrConfig(num_channels=9, image_size=config.image_size, num_countries=50)
    # model = GPTGuessrViT(gpt_guessr_config)
    model = GPTGuessrConvNeXt(num_countries=50)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    state_loss_function = nn.CrossEntropyLoss()
    
    train_encoder_loop(config, model, optimizer, state_loss_function, train_dataloader, eval_dataloader)

def train_encoder_loop(config, model, optimizer, state_loss_function, train_dataloader, eval_dataloader):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps, 
        log_with="tensorboard",
        project_dir=config.output_dir
    )
    if accelerator.is_main_process:
        accelerator.init_trackers("gptguessr")

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    global_step = 0
    
    for epoch in range(config.epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        
        model.train()
        for step, batch in enumerate(train_dataloader):
            images = batch['images']
            state = batch['state']
            coords = batch['coords']
            
            with accelerator.accumulate(model):
                pred_state, pred_coords = model(images)
                
                state_loss = state_loss_function(pred_state, state)
                dist_loss = coord_loss_function(pred_coords, coords) / 10000
                loss = state_loss + dist_loss
                accelerator.backward(loss)
                
                optimizer.step()
                optimizer.zero_grad()
                
            progress_bar.update(1)
            logs = {"train/dist_loss": dist_loss.detach().item(), "train/state_loss": state_loss.detach().item(), "train/loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
        
        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            model.eval()
            if (epoch + 1) % config.eval_epochs == 0 or epoch == config.epochs - 1:
                batch = next(iter(eval_dataloader))
                images = batch['images']
                state = batch['state']
                coords = batch['coords']
                pred_state, pred_coords = model(images)
                
                eval_state_loss = state_loss_function(pred_state, state).detach().item()
                eval_dist_loss = coord_loss_function(pred_coords, coords).detach().item()
                dist = torch.mean(haversine_distance(pred_coords, coords)).detach().item()
                
                logs = {"val/state_loss": eval_state_loss, "val/dist_loss": eval_dist_loss, "val/loss": eval_state_loss + eval_dist_loss, "val/dist": dist}
                accelerator.log(logs, step=global_step)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.epochs - 1:
                # accelerator.unwrap_model(model).save_pretrained(os.path.join(config.output_dir, f'guessr_{epoch + 1:03}.pth'))
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, os.path.join(config.output_dir, f'guessr_{epoch + 1:03}.pth'))
            

if __name__ == "__main__":
    train()
