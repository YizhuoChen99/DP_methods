import torch
from torch import nn

class DPPixModel(nn.Module):
    def __init__(self):
        super(DPPixModel, self).__init__()

    def forward(self, x):
        with torch.no_grad():
            # Break down into blocks
            b, c, h, w = x.size()
            block_size = 32
            x = x.reshape(b, c, block_size, h // block_size, block_size, w // block_size)

            # get the mean of each block
            x = x.mean(dim=(3, 5))
            # x: [b, c, block_size, block_size]

            # sample Laplace noise
            laplace = torch.distributions.Laplace(0, 0.1)
            noise = laplace.sample(x.size()).to(x.device)  # sample noise with the same shape as x_avg
            x = x + noise  # add noise to the input
            x = torch.clamp(x, 0, 1)

            x = x.unsqueeze(3).unsqueeze(5)  # [b, c, block_size, 1, block_size, 1]
            x = x.expand(b, c, block_size, h // block_size, block_size, w // block_size)  # [b, c, block_size, h // block_size, block_size, w // block_size]
            x = x.reshape(b, c, h, w)  # [b, c, h, w]

        return x
