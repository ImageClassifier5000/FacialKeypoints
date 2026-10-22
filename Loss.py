import torch
from torch import nn

class Loss(nn.Module):
    def __init__(self):
        self.mse_loss = nn.MSELoss()

    def forward(self,model_output,y):
        loss = 0
        for i in range(y.shape[0]):
            indices = torch.where(not torch.isnan(y[i]))
            output_without_nan_values = model_output[i][indices]
            labels_without_nan_values = y[i][indices]
            loss += self.mse_loss(output_without_nan_values,labels_without_nan_values)
        mean_loss = loss / y.shape[0]
        return mean_loss

