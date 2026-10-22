import torch
from torch import nn

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self,model_output,y):
        loss = 0
        for i in range(y.shape[0]):
            indices = torch.where(~torch.isnan(y[i]))
            output_without_nan_values = model_output[i][indices]
            labels_without_nan_values = y[i][indices]
            loss += self.mse_loss(output_without_nan_values,labels_without_nan_values)
        mean_loss = loss / y.shape[0]
        return mean_loss

from FacialKeypointsDataset import FacialKeypointsDataset
from KeypointsNetwork import KeypointsNetwork
import Train
import numpy as np
import pandas as pd
import torch
from torch import nn

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

dataset = FacialKeypointsDataset("./training/training.csv")
dataset[0][1]

loss_fn = Loss()

loss = loss_fn(torch.unsqueeze(dataset[0][1],dim=0), torch.unsqueeze(dataset[0][1],dim=0))
print(loss)