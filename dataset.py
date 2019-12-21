import open3d as o3d
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import random
#from utils import *

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size = n - pcd.shape[0])])
    return pcd[idx[:n]]
           
class ShapeNet(data.Dataset): 
    def __init__(self, train = True, npoints = 8192):
        if train:
            self.list_path = './data/train.list'
        else:
            self.list_path = './data/val.list'
        self.npoints = npoints
        self.train = train

        with open(os.path.join(self.list_path)) as file:
            self.model_list = [line.strip().replace('/', '_') for line in file]
        random.shuffle(self.model_list)
        self.len = len(self.model_list * 50)

    def __getitem__(self, index):
        model_id = self.model_list[index // 50]
        scan_id = index % 50
        def read_pcd(filename):
            pcd = o3d.io.read_point_cloud(filename)
            return torch.from_numpy(np.array(pcd.points)).float()
        if self.train:
            partial = read_pcd(os.path.join("./data/train/", model_id + '_%d_denoised.pcd' % scan_id))
        else:
            partial = read_pcd(os.path.join("./data/val/", model_id + '_%d_denoised.pcd' % scan_id))
        complete = read_pcd(os.path.join("./data/complete/", '%s.pcd' % model_id))       
        return model_id, resample_pcd(partial, 5000), resample_pcd(complete, self.npoints)

    def __len__(self):
        return self.len