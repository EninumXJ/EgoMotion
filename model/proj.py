import torch
import torch.nn as nn
import numpy as np
import torchvision
import sys
sys.path.append('..')
from utils.transforms import *
from torch.nn import functional as F

class Head(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_channels=512):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            nn.ReLU()
        )
       
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_channels, output_channels),
            nn.ReLU()
        )

    def forward(self, input):
        return self.fc2(self.fc1(input))
    
class Projector(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_channels=100):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            nn.ReLU()
        )
       
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_channels, output_channels),
            nn.ReLU()
        )

    def forward(self, input):
        return self.fc2(self.fc1(input))