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
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, num_ch, 7, 1, 3),
            nn.BatchNorm2d(num_ch),
            nn.ReLU()
        )

    def forward(self, inp):
        out = self.refine_net(inp)
        return out

# https://github.com/ethanluoyc/transporter-pytorch/blob/master/transporter.py
def spatial_softmax(features):
    features_reshape = features.reshape(features.shape[:-2] + (-1,))
    output = F.softmax(features_reshape, dim=-1)
    output = output.reshape(features.shape)
    return output

def compute_keypoint_location_mean(features):
    S_row = features.sum(-1)  # N, K, H
    S_col = features.sum(-2)  # N, K, W

    # N, K
    u_row = S_row.mul(torch.linspace(-1, 1, S_row.size(-1), dtype=features.dtype, device=features.device)).sum(-1)
    # N, K
    u_col = S_col.mul(torch.linspace(-1, 1, S_col.size(-1), dtype=features.dtype, device=features.device)).sum(-1)
    return torch.stack((u_row, u_col), -1) # N, K, 2

def gaussian_map(features, std=0.2):
    # features: (N, K, H, W)
    width, height = features.size(-1), features.size(-2)
    mu = compute_keypoint_location_mean(features)  # N, K, 2
    mu_y, mu_x = mu[:, :, 0:1], mu[:, :, 1:2]
    y = torch.linspace(-1.0, 1.0, height, dtype=mu.dtype, device=mu.device)
    x = torch.linspace(-1.0, 1.0, width, dtype=mu.dtype, device=mu.device)
    mu_y, mu_x = mu_y.unsqueeze(-1), mu_x.unsqueeze(-1)

    y = torch.reshape(y, [1, 1, height, 1])
    x = torch.reshape(x, [1, 1, 1, width])

    inv_std = 1 / std
    g_y = torch.pow(y - mu_y, 2)
    g_x = torch.pow(x - mu_x, 2)
    dist = (g_y + g_x) * inv_std**2
    g_yx = torch.exp(-dist)

    return g_yx

def transport(source_keypoints, target_keypoints, source_features, target_features):
    out = source_features
    for s, t in zip(torch.unbind(source_keypoints, 1), torch.unbind(target_keypoints, 1)):
        out = (1 - s.unsqueeze(1)) * (1 - t.unsqueeze(1)) * out + t.unsqueeze(1) * target_features
    return out

class Transporter(nn.Module):
    def __init__(self, encoder, keynet, refinenet, std=0.1):
        super().__init__()

        self.encoder = encoder
        self.key_net = keynet
        self.refine_net = refinenet
        self.std = std

    def forward(self, source_img, target_img):
        # source img
        source_features = self.encoder(source_img)
        source_kn = self.key_net(source_img)
        source_kn = spatial_softmax(source_kn)
        source_keypoints = gaussian_map(source_kn, self.std)

        # target img
        target_features = self.encoder(target_img)
        target_kn = self.key_net(target_img)
        target_kn = spatial_softmax(target_kn)
        target_keypoints = gaussian_map(target_kn, self.std)

        # transport
        transport_features = transport(source_keypoints.detach(),
                                    target_keypoints,
                                    source_features.detach(),
                                    target_features)

        # RefineNet
        out = self.refine_net(transport_features)
        return out

