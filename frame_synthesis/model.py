import torch
import torch.nn as nn
import torch.nn.functional as F

def meshgrid(height, width):
    x_t = torch.matmul(torch.ones(height, 1), torch.linspace(-1.0, 1.0, width).view(1, width))
    y_t = torch.matmul(torch.linspace(-1.0, 1.0, height).view(height, 1), torch.ones(1, width))

    grid_x = x_t.view(1, height, width)
    grid_y = y_t.view(1, height, width)

    return grid_x, grid_y

class VoxelFlow(nn.Module):
    def __init__(self, input_imgs=2, img_ch=3):
        super().__init__()

        self.input_imgs = input_imgs
        self.img_ch = img_ch
        self.input_mean = [0.5 * 255, 0.5 * 255, 0.5 * 255]
        self.input_std = [0.5 * 255, 0.5 * 255, 0.5 * 255]
        self.bn_momentum = 0.9997

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.max_pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(self.input_imgs*self.img_ch, 64, 5, 1, 2, bias=False)
        self.bn_c1 = nn.BatchNorm2d(64, momentum=self.bn_momentum)

        self.conv2 = nn.Conv2d(64, 128, 5, 1, 2, bias=False)
        self.bn_c2 = nn.BatchNorm2d(128, momentum=self.bn_momentum)

        self.conv3 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        self.bn_c3 = nn.BatchNorm2d(256, momentum=self.bn_momentum)

        self.bottleneck = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.bn_bn = nn.BatchNorm2d(256, momentum=self.bn_momentum)

        # in-channels -> add skip connections between layers
        self.deconv1 = nn.Conv2d(512, 256, 3, 1, 1, bias=False)
        self.bn_dc1 = nn.BatchNorm2d(256, momentum=self.bn_momentum)

        self.deconv2 = nn.Conv2d(384, 128, 5, 1, 2, bias=False)
        self.bn_dc2 = nn.BatchNorm2d(128, momentum=self.bn_momentum)

        self.deconv3 = nn.Conv2d(192, 64, 5, 1, 2, bias=False)
        self.bn_dc3 = nn.BatchNorm2d(64, momentum=self.bn_momentum)

        self.conv4 = nn.Conv2d(64, 3, 5, 1, 2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        inp = x
        device = inp.device
        # (H, W)
        inp_sz = tuple(x.size()[2:4])

        x = self.conv1(x)
        x = self.bn_c1(x)
        conv1 = self.relu(x)
        x = self.max_pool(conv1)
        
        x = self.conv2(x)
        x = self.bn_c2(x)
        conv2 = self.relu(x)
        x = self.max_pool(conv2)

        x = self.conv3(x)
        x = self.bn_c3(x)
        conv3 = self.relu(x)
        x = self.max_pool(conv3)

        x = self.bottleneck(x)
        x = self.bn_bn(x)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, conv3], 1)
        x = self.deconv1(x)
        x = self.bn_dc1(x)
        x = self.relu(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, conv2], 1)
        x = self.deconv2(x)
        x = self.bn_dc2(x)
        x = self.relu(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, conv1], 1)
        x = self.deconv3(x)
        x = self.bn_dc3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.tanh(x)

        flow = x[:, 0:2, :, :]
        mask = x[:, 2:3, :, :]

        grid_x, grid_y = meshgrid(*inp_sz)
        grid_x = grid_x.repeat([inp.size()[0], 1, 1]).to(device)
        grid_y = grid_y.repeat([inp.size()[0], 1, 1]).to(device)

        flow = 0.5*flow

        coor_x_1 = grid_x - flow[:, 0, :, :] * 2
        coor_y_1 = grid_y - flow[:, 1, :, :] * 2
        coor_x_2 = grid_x - flow[:, 0, :, :]
        coor_y_2 = grid_y - flow[:, 1, :, :]

        # designed to have 2 rbg-imgs as input
        out1 = F.grid_sample(inp[:, 0:3, :, :], torch.stack([coor_x_1, coor_y_1], 3), padding_mode='border', align_corners=True)
        out2 = F.grid_sample(inp[:, 3:6, :, :], torch.stack([coor_x_2, coor_y_2], 3), padding_mode='border', align_corners=True)

        mask = 0.5*(1.0+mask)
        mask = mask.repeat([1, 3, 1, 1])

        out = mask*out1+(1.0-mask)*out2

        return out

    def get_optim_policies(self):
        outs = []
        outs.extend(
            self.get_module_optim_policies(
                self,
                self.config,
                'model',
            ))
        return outs

    def get_module_optim_policies(self, module, config, prefix):
        weight = []
        bias = []
        bn = []

        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                ps = list(m.parameters())
                weight.append(ps[0])
                if len(ps) == 2:
                    bias.append(ps[1])
            elif isinstance(m, nn.BatchNorm2d):
                bn.extend(list(m.parameters()))

        return [
            {
                'params': weight,
                'lr_mult': 1,
                'decay_mult': 1,
                'name': prefix + " weight"
            },
            {
                'params': bias,
                'lr_mult': 2,
                'decay_mult': 0,
                'name': prefix + " bias"
            },
            {
                'params': bn,
                'lr_mult': 1,
                'decay_mult': 1,
                'name': prefix + " bn scale/shift"
            },
        ]