import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import os
import io
import h5py

#from piq import SSIMLoss
from piqa import SSIM

from torchsummary import summary

from dataloader import *
from models.baseline_UNet import *

def normalize_depth(depth, min_depth=None, max_depth=None, eps=1e-6):
    min_depth = depth.amin(dim=(2, 3), keepdim=True)
    max_depth = depth.amax(dim=(2, 3), keepdim=True)

    return (depth - min_depth) / (max_depth - min_depth + eps)

def abs_rel(pred, target, eps=1e-6):
    return torch.mean(torch.abs(pred - target) / (target + eps))

### HYPERPARAMETERS
EPOCHS = 50
BATCH_SIZE = 32

train_dataset = NYUDataLoader("/scratch/nchapre/nyudepthv2", split="train")
#val_dataset = NYUDataLoader("/scratch/nchapre/nyudepthv2", split="val")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
print(len(train_loader))
#val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

print(f"Number of batches loaded in train_loader: {len(train_loader)}")
print(f"Number of images in train_loader: {len(train_loader)*BATCH_SIZE}")
#print(f"Number of data samples loaded in val_loader: {len(val_loader)}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = UNet(n_channels=3, n_classes=1)
model = model.to(device)

summary(model, (3, 480, 640))

#loss_fn = SSIMLoss(data_range=1.0).to(device)
class SSIMLoss(SSIM):
    def forward(self, x, y):
        return 1. - super().forward(x, y)

loss_fn = SSIMLoss(n_channels=1, ).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
#loop = tqdm(train_loader)

train_loss_arr = []
val_loss_arr = []

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    val_loss = 0

    for images, depths in train_loader:

        #print(images.shape, depths.shape)
        #print(images[0])
       
        depths = depths.unsqueeze(1)

        images, depths = images.to(device), depths.to(device)
        images = images.float()
        
        preds = model(images)

        preds = normalize_depth(preds)
        depths = normalize_depth(depths)        

        loss = loss_fn(preds, depths)
        #loss = ssim_loss + 0.5*abs_rel(preds, depths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader.dataset)
    train_loss_arr.append(train_loss)
    print(f"Epoch number: {epoch+1}, total_train_loss: {train_loss}, avg_train_loss: {avg_train_loss}")

"""
    # Validation loop
    model.eval()
    with torch.no_grad():
        for images, depths in val_loader:
            depths = depths.unsqueeze(1)

            images, depths = images.to(device), depths.to(device)
            images = images.float()
	    
            preds = model(images)

            preds = normalize_depth(preds)
            depths = normalize_depth(depths)        

           # loss = loss_fn(preds, depths)
            ssim_loss = loss_fn(preds, depths)
            loss = ssim_loss + 0.5*abs_rel(preds, depths)
            val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_loss_arr.append(avg_val_loss)

    print(f"Epoch number: {epoch+1}, train_loss: {avg_train_loss}, val_loss: {avg_val_loss}")
"""

print(train_loss_arr)

def plot_loss(train_loss, val_loss):
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, 'b-', label='Train Loss')
    plt.plot(epochs, val_loss, 'r--', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#plot_loss(train_loss_arr, val_loss_arr)

torch.save(model.state_dict(), 'saved_models/baseline_UNet.pth')
