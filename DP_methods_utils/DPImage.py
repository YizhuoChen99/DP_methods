from argparse import Namespace
import torch
import torchvision.transforms as transforms
from torch import nn
from DP_methods_utils.models.psp import pSp

class DPImageModel(nn.Module):
    def __init__(self, model_path):
        super(DPImageModel, self).__init__()
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),])

        ckpt = torch.load(model_path, map_location='cpu')
        opts = ckpt['opts']
        opts['checkpoint_path'] = model_path
        opts['output_size'] = 1024

        opts = Namespace(**opts)
        self.net = pSp(opts)
        self.net.eval()

    def forward(self, x):
        with torch.no_grad():
            x = self.transform(x)
            for i in range(10):
                x = self.net(x, randomize_noise=False)
            # x = x * 0.5 + 0.5
            # x = torch.clamp(x, 0, 1)
        return x
