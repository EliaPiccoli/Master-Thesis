import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from kornia.losses import ssim_loss

# !! in tf channels is last dimension

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class VideoObjectSegmentationModel(nn.Module):
    def __init__(self, device="cpu", emb_size=512, K=20, depth=24, H=84, W=84):
        super().__init__()
        self.device = device

        self.of_reg_cur = 0.0
        self.of_reg_inc = 1e-5
        self.emb_size = emb_size
        self.K = K
        self.depth = depth
        self.H = H
        self.W = W
        self.final_conv_size = 7 * 7 * 64
        self.flow_c = 0.01
        
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
        self.fc_conv = nn.Linear(self.final_conv_size, emb_size)
        
        self.obj_trans = nn.Linear(emb_size, self.K*2)
        self.cam_trans = nn.Linear(emb_size, 2)

        self.fc_m1 = nn.Linear(emb_size, 21*21*self.depth)
        self.conv_m1 = nn.Conv2d(self.depth, self.depth, 3, 1, 1)
        self.conv_m2 = nn.Conv2d(self.depth, self.depth, 3, 1, 1)
        self.conv_m3 = nn.Conv2d(self.depth, self.K, 1, 1)
        self.upsample1 = nn.UpsamplingBilinear2d(size=(42, 42))
        self.upsample2 = nn.UpsamplingBilinear2d(size=(84, 84))

    def forward(self, input):
        # input shape = [ BS x C x H x W ]

        # Basic CNN
        x = self.cnn(input)
        x = self.relu(self.fc_conv(x))

        # Object Masks
        m = self.fc_m1(x)
        m = m.view((-1, self.depth, 21, 21))
        m = self.upsample1(m)

        # conv 5
        y = self.conv_m1(m)      # padding=1 => padding='same'
        m = self.relu(m + y)
        m = self.upsample2(m)

        # conv 6
        z = self.conv_m2(m)
        m = self.relu(m + z)

        # [ BS x K x H x W ]
        m = self.conv_m3(m)
        m = self.sigmoid(m)
        self.object_masks = m

        # Object Translation
        ot = self.obj_trans(x)
        # [ BS x K x 2 ]
        ot = torch.reshape(ot, (-1, self.K, 2))

        # Mesh Grid
        mesh_grid = self._create_mesh_grid().to(self.device)

        # Optical Flow
        # [ BS x K x 1 x H x W ]
        m_reshape = torch.unsqueeze(m, 2)

        # [ BS x K x 2 x 1 x 1 ]
        ot_reshape = torch.unsqueeze(torch.unsqueeze(ot, -1), -1)

        translation_masks = m_reshape * ot_reshape
        self.translation_masks = translation_masks

        # [ BS x 2 x H x W ]
        flow = torch.sum(translation_masks, 1)

        # [ BS x 2 x H*W ]
        flat_flow = torch.reshape(flow, (-1, 2, self.W*self.H))

        # Camera Translation
        # [ BS x 2 ]
        ct = self.cam_trans(x)
        ct = torch.unsqueeze(ct, -1)

        # add camera translation to flow
        # [ BS x 2 x H*W ]
        flat_flow = flat_flow + ct

        # Add in the default coordinates
        img_size_f = torch.from_numpy(self.flow_c * np.array([[self.H], [self.W]], dtype=np.float32)).to(self.device)
        img_size_flat_flow = img_size_f * flat_flow

        # [ BS x 2 x H*W]
        sampling_coords = torch.add(img_size_flat_flow, mesh_grid)

        # Computer transformed image
        y_s = sampling_coords[:, 0, :]
        ys_flat = torch.reshape(y_s, (-1,))

        x_s = sampling_coords[:, 1, :]
        xs_flat = torch.reshape(x_s, (-1,))

        # x1 -> x0
        x1 = input[:, 1, :, :]
        source_frames = torch.unsqueeze(x1, 1)

        # Interpolate
        out = self._interpolate(source_frames, xs_flat, ys_flat, (1, self.H, self.W))
        # [ BS x 1 x H x W ]
        out = torch.reshape(out, (input.size(0), 1, self.H, self.W))

        return out

    def compute_loss(self, x, x_):
        # DSSIM
        out_loss = ssim_loss(x, x_, 11)

        # L1 reg for translations masks
        of_loss_reg = torch.abs(self.translation_masks).mean(-1).mean(-1).mean(-1).mean(-1)

        loss = out_loss + self.of_reg_cur * of_loss_reg

        return loss.mean()

    def update_reg(self):
        # increase regularization
        self.of_reg_cur = min(self.of_reg_cur + self.of_reg_inc, 1)

    def _create_mesh_grid(self):
        x_lin = torch.linspace(0., self.W - 1., self.W)
        y_lin = torch.linspace(0., self.H - 1., self.H)

        grid_x, grid_y = torch.meshgrid(x_lin, y_lin)
        # meshgrid -> pytorch != tf :)
        grid_x = grid_x.t()
        grid_y = grid_y.t()

        grid_x = torch.reshape(grid_x, (1, -1))
        grid_y = torch.reshape(grid_y, (1, -1))

        grid = torch.cat([grid_y, grid_x], 0)

        return grid

    def _repeat(self, x, n_rep):
        a = torch.unsqueeze(torch.ones(n_rep, device=self.device), 1)
        rep = a.permute(1, 0)
        x = torch.reshape(x, (-1, 1)).to(torch.float32)
        y = torch.matmul(x, rep)
        y = torch.reshape(y, (-1,))
        return y

    # https://github.com/daviddao/spatial-transformer-tensorflow/blob/master/spatial_transformer.py
    def _interpolate(self, im, x, y, out_size):
        bs, c, h, w = im.shape

        x = x.to(torch.float32)
        y = y.to(torch.float32)
        out_h = out_size[1]
        out_w = out_size[2]
        zero = torch.zeros([], dtype=torch.int32, device=self.device)
        max_y = h - 1
        max_x = w - 1

        # sampling
        x0 = torch.floor(x)
        x0 = x0.to(torch.int32)
        x0 = torch.clamp(x0, zero, max_x)
        unclip_x1 = x0 + 1
        x1 = torch.clamp(unclip_x1, zero, max_x)

        y0 = torch.floor(y)
        y0.to(torch.int32)
        y0 = torch.clamp(y0, zero, max_y)
        unclip_y1 = y0 + 1
        y1 = torch.clamp(unclip_y1, zero, max_y)

        dim2 = w
        dim1 = w*h
        z = torch.arange(0, bs, device=self.device)*dim1
        base = self._repeat(z, out_h*out_w)
        base_y0 = base + y0*dim2
        base_y1 = base + y1*dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        im_flat = torch.reshape(im, (-1, c))
        im_flat = im_flat.to(torch.float32)
        Ia = im_flat[idx_a.to(torch.long)]
        Ib = im_flat[idx_b.to(torch.long)]
        Ic = im_flat[idx_c.to(torch.long)]
        Id = im_flat[idx_d.to(torch.long)]

        # unclip_x1 = torch.clamp(unclip_x1, zero, max_x + 1)
        # unclip_y1 = torch.clamp(unclip_y1, zero, max_y + 1)
        
        x = torch.clamp(x, 0., float(max_x))
        y = torch.clamp(y, 0., float(max_y))

        # calculate interpolated values
        x0_f = x0.to(torch.float32)
        x1_f = x1.to(torch.float32)     # ?
        y0_f = y0.to(torch.float32)
        y1_f = y1.to(torch.float32)     # ?
        wa = torch.unsqueeze((x1_f-x) * (y1_f-y), 1)
        wb = torch.unsqueeze((x1_f-x) * (y-y0_f), 1)
        wc = torch.unsqueeze((x-x0_f) * (y1_f-y), 1)
        wd = torch.unsqueeze((x-x0_f) * (y-y0_f), 1)
        output = wa*Ia + wb*Ib + wc*Ic + wd*Id

        return output