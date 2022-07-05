import torch
import time
from model import VoxelFlow

device = "cuda:2"
model = VoxelFlow()
model = model.to(device)
model.train()

opt = torch.optim.Adam(model.parameters())
L = torch.nn.MSELoss()

for i in range(10):
    opt.zero_grad()
    inp = torch.randn(32, 6, 256, 256, device=device)
    exp = torch.randn(32, 3, 256, 256, device=device)
    print("IN:", inp.shape)
    out = model(inp)
    print("OUT:", out.shape)
    loss = L(out, exp)
    loss.backward()
    opt.step()

# TODO:
#   1. create dataset and data loader - train/validation
#   2. set up training cycle
#       - optimizer
#       - loss
#       - eval
#   3. set up wandb
#   4. train and test model