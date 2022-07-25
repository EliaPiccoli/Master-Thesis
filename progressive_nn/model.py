import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PNNCol(nn.Module):
    '''
       [0] Conv2D
       [1] Conv2D
       [2] Conv2D
       [3] Linear
       [4] Linear
    '''
    def __init__(self, col_id, num_channels, n_actions, input_size, base_skills, hidden_size=256):
        super().__init__()

        self.col_id = col_id
        self.num_layers = 5

        # init all base skills and freeze them
        self.skills = nn.ModuleList()
        self.skills_name = []
        # skills order: state-representation, video-segmentation, keypoints
        for name, skill in base_skills:
            self.skills_name.append(name)
            self.skills.append(skill)
            self.skills[-1].eval()
            for param in self.skills[-1].parameters():
                param.requires_grad = False

        # adapters for skills (hardcoded)
        self.adapter_segmentation = nn.Conv2d(21, 21, 5, 5)
        self.adapter_keypoints = nn.Conv2d(132, 132, 6)
        self.relu = nn.ReLU()

        # init normal model, lateral connection, adapter layer and alpha
        self.w = nn.ModuleList()
        self.u = nn.ModuleList()
        self.v = nn.ModuleList()
        self.alpha = nn.ModuleList()

        # define nn model
        self.w.append(
            nn.Conv2d(num_channels, 64, 3, 1, 1),
        )
        self.w.extend([
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.Conv2d(32, 32, 3, 1, 1)
        ])
        conv_out_size = self._get_conv_out(input_size)
        self.w.append(
            nn.Linear(conv_out_size, hidden_size)
        )
        self.w.append(
            nn.Linear(hidden_size, n_actions)
        )
        
        # lateral connections, adapter and alpha for col_id != 0
        # v[col][layer]
        for i in range(self.col_id):
            # adapter
            self.v.append(nn.ModuleList)
            self.v[i].append(nn.Identity())
            self.v[i].extend([
                nn.Conv2d(32, 1, 1),
                nn.Conv2d(32, 1, 1)
            ])
            self.v[i].append(nn.Linear(conv_out_size, conv_out_size))
            self.v[i].append(nn.Linear(hidden_size, hidden_size))

            # alpha
            self.alpha.append(nn.ModuleList())
            self.alpha[i].append(
                nn.Parameter(torch.Tensor(1), requires_grad=False)
            )
            self.alpha[i].extend([
                nn.Parameter(torch.Tensor(np.array(np.random.choice([1., 1e-1, 1e-2]))))
                for _ in range(self.num_layers - 1)
            ])

            # lateral connection
            self.u.append(nn.ModuleList())
            self.u[i].append(nn.Identity())
            self.u[i].extend([
                nn.Conv2d(1, 32, 3, 2, 1),
                nn.Conv2d(1, 32, 3, 2, 1)
            ])
            self.u[i].append(nn.Linear(conv_out_size, hidden_size))
            self.u[i].append(nn.Linear(hidden_size, n_actions))

    def apply_skills(self, x):
        # use skills to elaborate the input
        state_out = None
        video_out = None
        key_out = None
        with torch.no_grad():
            for i in range(len(self.skills)):
                if self.skills_name[i] == "state-representation":
                    # print("--------- STATE SKILL ---------")
                    # print("INP_SZ:", x.shape)
                    # resize to bigger image
                    rx = F.interpolate(x, (160, 210), mode='bilinear', align_corners=True)
                    # print("RX:", rx.shape)
                    # grayscale
                    gray_rx = rx[:, 0, :, :]*0.2125 + rx[:, 1, :, :]*0.7154 + rx[:, 2, :, :]*0.0721
                    gray_rx = torch.unsqueeze(gray_rx, 1)
                    # print("INP_RSZ:", gray_rx.shape, gray_rx.device)
                    o = self.skills[i](gray_rx)
                    # print("OUT_SZ:", o.shape)
                    # from linear to img just reshape
                    o = torch.reshape(o, (-1, 16, 16))
                    o = torch.unsqueeze(o, 1)
                    state_out = o
                    # print("STATE_OUT:", state_out.shape)
                elif self.skills_name[i] == 'video-segmentation':
                    # print("--------- VIDEO SKILL ---------")
                    # print("INP_SZ:", x.shape)
                    # grayscale & normalized input
                    gray_x = x[:, 0, :, :]*0.2125 + x[:, 1, :, :]*0.7154 + x[:, 2, :, :]*0.0721
                    gray_x = torch.unsqueeze(gray_x, 1)
                    norm_gray_x = gray_x / 255.
                    # print("NGX:", norm_gray_x.shape)
                    inp = torch.cat([norm_gray_x, norm_gray_x], 1)
                    # print("INP:", inp.shape)
                    o = self.skills[i](inp)
                    # print("O:", o.shape)
                    # print("OBJM:", self.skills[i].object_masks.shape)
                    video_out = torch.cat([o, self.skills[i].object_masks], 1)
                    # print("VIDEO_OUT:", video_out.shape)
                elif self.skills_name[i] == "keypoints":
                    # print("--------- KEYPOINTS SKILL ---------")
                    # print("INP_SZ:", x.shape)
                    o_enc = self.skills[i].encoder(x)
                    o_key = self.skills[i].key_net(x)
                    # print("OE:", o_enc.shape)
                    # print("OK:", o_key.shape)
                    o = torch.cat([o_enc, o_key], 1)
                    key_out = o
                    # print("KEYPOINTS_OUT:", key_out.shape)
                else:
                    raise NotImplemented(f"{self.skills_name[i]} not implemented")

            # print("--------- MERGE SKILLS ---------")

            # print(state_out.shape)
            # print(video_out.shape)
            # print(key_out.shape)

        adapt_video_out = self.relu(self.adapter_segmentation(video_out))
        adapt_key_out = self.relu(self.adapter_keypoints(key_out))

        # real input is the concatenation of all the input skills
        x = torch.cat([state_out, adapt_key_out, adapt_video_out], 1)
        # print("SKILL_OUT:", x.shape)

        return x

    def forward(self, x, pre_out):
        # IN [ BS x 84 x 84 x 3]
        # print(x.shape)

        # [ BS x 3 x 84 x 84]
        x = x.permute(0, 3, 1, 2)
        x = self.apply_skills(x)

        # # print("--------- AGENT ---------")
        # put a placeholder to occupy the first layer spot
        next_out, w_out = [torch.zeros(x.shape)], x

        output = None
        # all layers[:-1]
        for i in range(self.num_layers - 1):
            if i == self.num_layers - 2:
                # Flatten
                w_out = w_out.view(w_out.size(0), -1)
                # Flatten all the previous output too
                for k in range(self.col_id):
                    pre_out[k][i] = pre_out[k][i].view(pre_out[k][i].size(0), -1)

            # model out
            w_out = self.w[i](w_out)
            # previous col out
            u_out = [
                self.u[k][i](self.relu(self.v[k][i](self.alpha[k][i]*pre_out[k][i])))
                if self.col_id != 0 else torch.zeros(w_out.shape)
                for k in range(self.col_id)
            ]

            w_out = self.relu(w_out + sum(u_out))
            next_out.append(w_out)

        # output layer
        # model
        output = self.w[-1](w_out)
        # prev col
        prev_out = [
            self.u[k][-1](self.relu(self.v[k][-1](self.alpha[k][-1]*pre_out[k][-1])))
            if self.col_id != 0 else torch.zeros(output.shape)
            for k in range(self.col_id)
        ]

        return output + sum(prev_out), next_out

    def _get_conv_out(self, shape):
        output = torch.zeros(1, *shape)
        for i in range(self.num_layers - 2):
            output = self.w[i](output)
        return int(np.prod(output.size()))

    def train(self, mode=True):
        super().train(mode)
        for skill in self.skills:
            skill.eval()


class PNN(nn.Module):
    def __init__(self, allcol):
        super().__init__()

        self.columns = nn.ModuleList()
        for col in allcol:
            self.columns.append(col)

        self.freeze()

    def freeze(self):
        if len(self.columns) == 1:
            return
        
        for i in range(len(self.columns) - 1):
            self.columns[i].eval()
            for param in self.columns[i].parameters():
                param.requires_grad = False

    def forward(self, x):
        output, next_out = None, []

        for i in range(len(self.columns)):
            # print(f"C{i}")
            output, col_out = self.columns[i](x, next_out)
            # print(f"C{i} - OUT:", output.shape)
            # print(f"C{i} - COL_OUT", len(col_out))
            next_out.append(col_out)

        return output
    
    def train(self, mode=True):
        super().train(mode)
        self.freeze()