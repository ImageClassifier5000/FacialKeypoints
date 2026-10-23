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
(train_dataset,second_dataset) = dataset.split((80,20))
(test_dataset, validation_dataset) = second_dataset.split((50,50))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = KeypointsNetwork().to(device)
model.load_state_dict(
    torch.load("model_weights.pth", map_location=device, weights_only=True)
)

# constants

train_dataloader = DataLoader(
    train_dataset,
    batch_size=100,
    shuffle=True,
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=True,
)
learning_rate = 5e-4
epochs = 3
loss_fn = Loss()

# loss_fn
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    Train.train_loop(train_dataloader, model, loss_fn, optimizer, device)
    Train.test_loop(test_dataloader, model, loss_fn, device)
    torch.save(model.state_dict(), 'model_weights.pth')
print("Done!")