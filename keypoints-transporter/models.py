import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 7, 1, (1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, (1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, (1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, (1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

    def forward(self, inp):
        out = self.cnn(inp)

        return out

class KeyNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp):
        pass

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp):
        pass

class Transporter(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp):
        pass