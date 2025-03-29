import torch
from torch import nn

class SnowModel(nn.Module):
    def __init__(self):
        super(SnowModel, self).__init__()

    def forward(self, x):
        with torch.no_grad():
            mask = torch.rand_like(x) < 0.9
            snow = torch.zeros_like(x)
            x = torch.where(mask, snow, x)

        return x
