import torch
import torch.nn as nn
import torch.nn.functional as F

# !! in tf channels is last dimension

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class VideoObjectSegmentationModel(nn.Module):
    def __init__(self, K=20, depth=24):
        super().__init__()
        
        self.K = K
        self.depth = depth
        self.final_conv_size = 7 * 7 * 64 # check
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            Flatten()
        )
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc_conv = nn.Linear(self.final_conv_size, 512)
        self.obj_trans = nn.Linear(512, self.K*2)

        self.fc_m1 = nn.Linear(512, 21*21*self.depth)
        self.conv_m1 = nn.Conv2d(self.depth, self.depth, 3, 1)
        self.conv_m2 = nn.Conv2d(self.depth, self.depth, 3, 1)
        self.conv_m3 = nn.Conv2d(self.depth, self.K, 1, 1)
        self.upsample1 = nn.UpsamplingBilinear2d(size=(42, 42))
        self.upsample2 = nn.UpsamplingBilinear2d(size=(84, 84))

    def forward(self, input):
        # input shape = [ BS x C x H x W ]

        # Basic CNN
        x = self.cnn(input)
        x = self.relu(self.fc_conv(x))

        # Object & Camera Translation
        ot = self.obj_trans(x)
        # [ BS x K x 2 ]
        ot = torch.reshape(ot, (-1, self.K, 2))

        # Object Masks
        m = self.fc_m1(x)
        m = torch.reshape(m, (-1, self.depth, 21, 21))
        m = self.upsample1(m)

        # conv pad -> same
        y = F.pad(m, (1, 1, 1, 1))
        y = self.conv_m1(y)
        m = self.relu(m + y)
        m = self.upsample2(m)

        # conv pad -> same
        z = F.pad(m, (1, 1, 1, 1))
        z = self.conv_m2(z)
        m = self.relu(m + z)

        # [ BS x K x H x W ]
        m = self.conv_m3(m)
        m = self.sigmoid(m)

        # Optical Flow
        # [ BS x K x 2 x 1 x 1 ]
        ot_reshape = torch.unsqueeze(torch.unsqueeze(ot, -1), -1)

        # [ BS x K x 1 x H x W ]
        m_reshape = torch.unsqueeze(m, 2)

        out = ot_reshape * m_reshape

        # [ BS x 2 x H x W]
        out = torch.sum(out, 1)

        return out