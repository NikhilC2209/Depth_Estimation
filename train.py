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
from tqdm import tqdm

from torchsummary import summary

import sys

from dataloader import *
from models.baseline_UNet import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = UNet(n_channels=3, n_classes=1)
model = model.to(device)

summary(model, (3, 480, 640))

loss_fn = nn.BCEWithLogitsLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
#loop = tqdm(train_loader)

for images, depths in train_loader:

    #print(images.shape, depths.shape)
    #print(images[0])
    
    images, depths = images.to(device), depths.to(device)
    images = images.float()
    
    preds = model(images)
    loss = loss_fn(preds, depths)