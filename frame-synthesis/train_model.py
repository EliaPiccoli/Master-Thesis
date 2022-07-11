import wandb
import os
import torch
import sys
import numpy as np
from model import VoxelFlow
from dataset import Dataset, Sampler
from torch.utils.data import DataLoader
from kornia.losses import SSIMLoss

if len(sys.argv) < 3:
    print(f"Usage: python {sys.argv[0]} <env> <gpu-device>")
    exit()
ENV = sys.argv[1]
gpu = sys.argv[2]

device = torch.device(gpu if torch.cuda.is_available() else "cpu")

data_path = f"data/{ENV}"
NUM_EPS = 1000
MAX_EP_LEN = 100
img_ch = 3
img_sz = (256, 256)
batch_size = 32

eps = np.arange(NUM_EPS)
np.random.shuffle(eps)
split_idx = int(NUM_EPS*0.8)
train_idxs = eps[:split_idx]
val_idxs = eps[split_idx:NUM_EPS]

dataset_ts = Dataset(data_path, train_idxs, MAX_EP_LEN)
train_load = DataLoader(dataset_ts, batch_size, shuffle=True, num_workers=8)

dataset_vs = Dataset(data_path, val_idxs, MAX_EP_LEN)
val_load = DataLoader(dataset_vs, batch_size, num_workers=8)

lr = 0.0001
w_decay = 1e-4
max_epoch = 400*len(train_load)

wandb.init(project="thesis", entity="epai", tags=["FrameSynthesis"])
wandb.config.update({
        "env": ENV,
        "num_eps": NUM_EPS,
        "max_ep_len": MAX_EP_LEN,
        "image-channels": img_ch,
        "image-size": img_sz,
        "steps": max_epoch,
        "batch-size": batch_size,
        "lr": lr,
        "weight-decay": w_decay,
    })

model = VoxelFlow()
model = model.to(device)

opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)
L = torch.nn.MSELoss().cuda()
E = SSIMLoss(11) # it computes DSSIM :)

best_eval = 0.0
for e in range(max_epoch):
    model.train()
    train_losses = []
    for i, (inp, target) in enumerate(train_load):
        opt.zero_grad()
        
        inp = inp.to(device)
        target = target.to(device)
        out = model(inp)

        loss = L(out, target)
        train_losses.append(loss.item())

        loss.backward()
        opt.step()
    avg_train_loss = sum(train_losses)/len(train_losses)

    val_metrics = []
    with torch.no_grad():
        model.eval()
        for i, (inp, target) in enumerate(val_load):
            inp = inp.to(device)
            target = target.to(device)
            out = model(inp)

            eval_ = 1 - E(out, target)
            val_metrics.append(eval_.item())
    avg_ssim = sum(val_metrics)/len(val_metrics)

    if avg_ssim > best_eval:
        best_eval = avg_ssim
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, ENV + '.pt'))

    # print(f"{e} - tr: {avg_train_loss} val: {avg_ssim} best: {best_eval}")
    wandb.log({
        "tr_loss": avg_train_loss,
        "eval" : avg_ssim,
        "best_eval": best_eval
    }, step=e)

torch.save(model.state_dict(), os.path.join(wandb.run.dir, ENV + '_final.pt'))