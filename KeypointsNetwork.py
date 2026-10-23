import torch
from torch import nn

class KeypointsNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolutional_network = nn.Sequential(
        nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,padding=1,stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
        nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,padding=1,stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
        nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=1,stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
        nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1,stride=1),
        nn.ReLU()
        )
        self.endprocessing = nn.Sequential(
            nn.modules.flatten.Flatten(start_dim=1,end_dim=-1),
            nn.Linear(in_features=12*12*32,out_features=30),
            nn.ReLU()
        )
    def forward(self,x):
        x = self.convolutional_network(x)
        return self.endprocessing(x)