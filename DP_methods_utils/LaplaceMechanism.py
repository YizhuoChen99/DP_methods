import torch
from torch import nn

class LaplaceMechanismModel(nn.Module):
    def __init__(self):
        super(LaplaceMechanismModel, self).__init__()

    def forward(self, x):
        with torch.no_grad():
            # sample Laplace noise
            loc = 0.0  # mean of the Laplace distribution
            scale = 2.0  # diversity (scale) of the Laplace distribution
            laplace = torch.distributions.Laplace(loc, scale)
            noise = laplace.sample(x.size()).to(x.device)  # sample noise with the same shape as x
            x = x + noise  # add noise to the input
            # x = torch.clamp(x, 0, 1)  # clamp the output to [0, 1]
        return x
