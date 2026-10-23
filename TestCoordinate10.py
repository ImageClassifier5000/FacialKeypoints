from FacialKeypointsDataset import FacialKeypointsDataset
from KeypointsNetwork import KeypointsNetwork
import Train
import numpy as np
import pandas as pd
import torch
from torch import nn
from Loss import Loss

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

dataset = FacialKeypointsDataset("./training/training.csv")

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = KeypointsNetwork().to(device)
model.load_state_dict(
    torch.load("model_weights.pth", map_location=device, weights_only=True)
)
model = KeypointsNetwork()
model.load_state_dict(
        torch.load("model_weights.pth", map_location=device, weights_only=True)
    )


with torch.no_grad():
        index = 0
        problematic_count = 0
        for X, y in dataloader:
            pred = model(X)
            problematic_variable = torch.flatten(pred[::,20])
            amount_of_problematic_values = len(torch.nonzero(problematic_variable))
            if amount_of_problematic_values != 32:
                print(f"Nach {problematic_count} problematischen Iterationen: {amount_of_problematic_values}")
            else:
                problematic_count += 1