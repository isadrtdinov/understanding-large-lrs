import torch
import numpy as np
from torch import nn
from torchvision import transforms

class PreActLeNet(nn.Module):
    def __init__(self, scale=25, num_classes=10):
        super().__init__()
        
        self.scale = scale
        self.sizes = self.scale * np.array([1, 2, 16])
        self.fc_1_size = 16 * self.sizes[1]
        
        self.pool = nn.Sequential(
            nn.Conv2d(1, self.sizes[0], 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.sizes[0], self.sizes[1], 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.clf = nn.Sequential(
            nn.Linear(self.fc_1_size, self.sizes[2]),
            nn.ReLU(),
            nn.Linear(self.sizes[2], num_classes)
        )

    def forward(self, x):
        x = self.pool(x)
        x = x.view(-1, self.fc_1_size)
        return self.clf(x)

class LeNet:
    base = PreActLeNet
    args = []
    kwargs = {}
    transform_train = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])
