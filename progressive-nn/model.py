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
    def __init__(self, col_id, num_channels, n_actions, input_size, hidden_size=256):
        super().__init__()

        self.col_id = col_id
        self.num_layers = 5

        # init normal model, lateral connection, adapter layer and alpha
        self.w = nn.ModuleList()
        self.u = nn.ModuleList()
        self.v = nn.ModuleList()
        self.alpha = nn.ModuleList()

        # define nn model
        self.w.append(
            nn.Conv2d(num_channels, 12, 8, 4, 1),
        )
        self.w.extend([
            nn.Conv2d(12, 12, 4, 2, 1),
            nn.Conv2d(12, 12, (3, 4), 1, 1)
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
                nn.Conv2d(12, 1, 1),
                nn.Conv2d(12, 1, 1)
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
                nn.Conv2d(1, 12, 4, 2, 1),
                nn.Conv2d(1, 12, (3, 4), 1, 1)
            ])
            self.u[i].append(nn.Linear(conv_out_size, hidden_size))
            self.u[i].append(nn.Linear(hidden_size, n_actions))

    def forward(self, x, pre_out):
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
                self.u[k][i](nn.ReLU(self.v[k][i](self.alpha[k][i]*pre_out[k][i])))
                if self.col_id != 0 else torch.zeros(w_out.shape)
                for k in range(self.col_id)
            ]

            w_out = nn.ReLU(w_out + sum(u_out))
            next_out.append(w_out)

        # output layer
        # model
        output = self.w[-1](w_out)
        # prev col
        prev_out = [
            self.u[k][-1](nn.ReLU(self.v[k][-1](self.alpha[k][-1]*pre_out[k][-1])))
            if self.col_id != 0 else torch.zeros(output.shape)
            for k in range(self.col_id)
        ]

        return output + sum(prev_out), next_out

    def _get_conv_out(self, shape):
        output = torch.zeros(1, *shape)
        for i in range(self.nlayers - 2):
            output = self.w[i](output)
        return int(np.prod(output.size()))