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

from geo_guessr_dataset import GPTGuessrDataset
from gpt_guessr import GPTGuessr, GPTGuessrConfig

from accelerate import Accelerator

from dataclasses import dataclass

@dataclass
class TrainingConfig:
    rgb = True
    image_size = 512  # the generated image resolution
    train_batch_size = 12
    eval_batch_size = 12  # how many images to sample during evaluation
    num_dataloader_workers = 12  # how many subprocesses to use for data loading
    epochs = 40
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    save_image_epochs = 5
    save_model_epochs = 5
    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = 'eg3d-latent-diffusion'
    
    eg3d_model_path = 'eg3d/eg3d_model/ffhqrebalanced512-128.pkl'
    eg3d_latent_vector_size = 512
    
    data_dir = 'data/'
    df_file = 'dataset.df'

    overwrite_output_dir = True
    seed = 0

def haversine_distance(p, t):
    # Convert latitude and longitude to radians
    p, t = torch.tensor([p, t], dtype=torch.float) * np.pi / 180
    
    # Compute the haversine formula
    a = torch.sin((t[:, 0]-p[:, 0])/2)**2 + torch.cos(p) * torch.cos(t) * torch.sin((t[:, 1]-p[:, 1])/2)**2
    c = 2 * torch.arcsin(torch.sqrt(a))
    
    # Earth's radius (mean radius = 6,371km)
    earth_radius = 6371
    
    return earth_radius * c


def coord_loss_function(p, t):
    dist = haversine_distance(p, t)
    return F.SmoothL1Loss(dist, torch.zeros_like(dist))


def train():    
    config = TrainingConfig()
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            normalize,
        ])

    dataset = GPTGuessrDataset(df_file=config.df_file, data_dir=config.data_dir, transform=preprocess)

    train_size = int(len(dataset) * 0.1)
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size], generator=torch.Generator().manual_seed(42))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_dataloader_workers)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=config.eval_batch_size, shuffle=True, num_workers=config.num_dataloader_workers)
    
    print(f"Loaded Dataloaders")
    print(f"Training on {len(train_dataset)} images, evaluating on {len(eval_dataset)} images")
    
    ### TRAIN GPTGuessr ###
    gpt_guessr_config = GPTGuessrConfig()
    model = GPTGuessr(gpt_guessr_config)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    country_loss_function = nn.CrossEntropy()
    coord_loss_function = coord_loss_function
    
    train_encoder_loop(config, model, optimizer, country_loss_function, coord_loss_function, train_dataloader, eval_dataloader)

def train_encoder_loop(config, model, optimizer, country_loss_function, coord_loss_function, train_dataloader, eval_dataloader):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps, 
        log_with="tensorboard",
        logging_dir=os.path.join(config.output_dir, "logs")
    )
    if accelerator.is_main_process:
        accelerator.init_trackers("eg3d_li_encoder")

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    global_step = 0
    
    for epoch in range(config.encoder_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        
        model.train()
        
        for step, batch in enumerate(train_dataloader):
            images = batch['images']
            country = batch['country']
            coords = batch['coords']
            
            with accelerator.accumulate(model):
                pred_country, pred_coords = model(images)
                
                country_loss = country_loss_function(pred_country, country)
                coord_loss = coord_loss_function(pred_coords, coords)
                loss = country_loss + coord_loss
                accelerator.backward(loss)
                
                optimizer.step()
                optimizer.zero_grad()
                
            progress_bar.update(1)
            logs = {"train_coord_loss": loss.detach().item(), "train_country_loss": country_loss.detach().item(), "train_loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
        
        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            model.eval()
            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.encoder_epochs - 1:
                batch = next(iter(eval_dataloader))
                images = batch['images']
                country = batch['country']
                coords = batch['coords']
                pred_country, pred_coords = model(images)
                eval_country_loss = country_loss_function(pred_country, country).detatch().item()
                eval_coord_loss = coord_loss_function(pred_coords, coords).detatch().item()
                dist = haversine_distance(pred_coords, coords)
                logs = {"val_country_loss": eval_country_loss, "val_coord_loss": eval_coord_loss, "val_loss": eval_country_loss + eval_coord_loss, "val_dist": dist}
                accelerator.log(logs, step=global_step)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.encoder_epochs - 1:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, os.path.join(config.output_dir, f'encoder_{epoch + 1}.pth'))
                


if __name__ == "__main__":
    train()
