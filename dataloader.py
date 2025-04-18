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

class NYUDataLoader(Dataset):
    def __init__(self, dir, split='train', transform=None, depth_transform=None):
        super(NYUDataLoader, self).__init__()
        
        self.dir = os.path.join(dir, split)
        self.h5_pathnames = self.populate_pathnames()
    
    def __len__(self):
        return len(self.h5_pathnames)

    def __getitem__(self, idx):
        h5_path = self.h5_pathnames[idx]

        #print(f"this is our h5 path: {h5_path}")
        
        with open(h5_path, "rb") as fp:
            rgb, depth = self.h5_loader(fp.read())

        #print(rgb.shape, depth.shape)
        
        return rgb, depth

    def populate_pathnames(self):

        pathnames = []
        
        for folder in os.listdir(self.dir):
            scene_path = os.path.join(self.dir, folder)
            if os.path.isdir(scene_path):
                for fname in os.listdir(scene_path):
                    if fname.endswith('.h5'):
                        pathnames.append(os.path.join(scene_path, fname))

        return pathnames
    
    def h5_loader(self, bytes_stream):
        f = io.BytesIO(bytes_stream)
        h5f = h5py.File(f, "r")
        rgb = np.array(h5f["rgb"])
        #rgb = np.transpose(rgb, (1, 2, 0))
        depth = np.array(h5f["depth"])
        return rgb, depth
    
    def h5_check(self):
        dirs = os.listdir(self.dir)
        h5_dir = os.path.join(self.dir, dirs[0])
        
        return h5_dir, os.listdir(h5_dir)

    def load_one_sample(self):
        h5_dir, h5_files = self.h5_check()

        h5_path = os.path.join(h5_dir, h5_files[0])
        print(h5_path)
        with open(h5_path, "rb") as fp:
            rgb, depth = self.h5_loader(fp.read())

        print(rgb.shape)
        print(depth.shape)

        return rgb, depth

    def show_one_sample(self):
        rgb, depth = self.load_one_sample()

        batch = np.hstack([rgb, self.colored_depthmap(depth)])

        plt.imshow(batch.astype("uint8"))
        plt.axis("off")
        plt.show()
    
    def colored_depthmap(self, depth, d_min=None, d_max=None):

        cmap = plt.cm.viridis
        
        if d_min is None:
            d_min = np.min(depth)
        if d_max is None:
            d_max = np.max(depth)
        depth_relative = (depth - d_min) / (d_max - d_min)
        return 255 * cmap(depth_relative)[:,:,:3] # H, W, C

    def show_depthmap(self, depth_map):
       if not isinstance(depth_map, np.ndarray):
           depth_map = np.array(depth_map)
       if depth_map.ndim == 3:
           depth_map = depth_map.squeeze()
    
       d_min = np.min(depth_map)
       d_max = np.max(depth_map)
       depth_map = colored_depthmap(depth_map, d_min, d_max)
    
       plt.imshow(depth_map.astype("uint8"))
       plt.axis("off")
       plt.show()

if __name__ == '__main__':

	train_dataset = NYUDataLoader("D:\\fastdepth\\nyudepthv2" ,split="train")
	train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

	batch = next(iter(train_loader))
	print(batch.shape)