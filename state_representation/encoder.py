import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_channels, feature_size):
        super().__init__()
        self.feature_size = feature_size
        self.input_channels = input_channels
        self.final_conv_size = 64 * 9 * 6
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Linear(self.final_conv_size, self.feature_size)

    def forward(self, inputs):
        x = self.cnn(inputs)
        x = x.view(x.size(0), -1)
        return self.fc(x)