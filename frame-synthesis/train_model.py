import torch
import sys
from model import VoxelFlow

if len(sys.argv) < 3:
    print(f"Usage: python {sys.argv[0]} <env> <gpu-device>")
    exit()
ENV = sys.argv[1]
gpu = sys.argv[2]

device = torch.device(gpu if torch.cuda.is_available() else "cpu")

data_path = f"data/{ENV}"
NUM_EPS = 100
MAX_EP_LEN = 100
img_ch = 3
img_sz = (256, 256)

batch_size = 32
# lr = ...







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