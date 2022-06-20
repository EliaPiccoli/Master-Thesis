import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, inp_ch=3):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(inp_ch, 32, 7, 1, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

    def forward(self, inp):
        out = self.encoder(inp)
        return out

class KeyNet(nn.Module):
    def __init__(self, inp_ch=3, K=1):
        super().__init__()

        self.keynet = nn.Sequential(
            nn.Conv2d(inp_ch, 32, 7, 1, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.reg = nn.Conv2d(128, K, 1)

    def forward(self, inp):
        x = self.keynet(inp)
        out = self.reg(x)
        return out

class RefineNet(nn.Module):
    def __init__(self, num_ch):
        super().__init__()

        self.refine_net = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.Conv2d(32, num_ch, 7, 1, 3),
            nn.BatchNorm2d(num_ch),
            nn.ReLU()
        )

    def forward(self, inp):
        out = self.refine_net(inp)
        return out

class Transporter(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp):
        pass