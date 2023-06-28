import os
import torch
import torch.utils.data

from glob import glob


class EgoDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, config_path, transform=None, ext='jpg'):
        self.dataset_path = dataset_path
        self.config_path = config_path

        
    def __getitem__(self, idx):

    
    def __len__(self):


    def 