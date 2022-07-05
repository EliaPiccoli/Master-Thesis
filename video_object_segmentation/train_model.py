import numpy as np
import torch
import wandb
import os
import sys
import random
import torch.nn.functional as F
from model import VideoObjectSegmentationModel
from dataset import Dataset

if len(sys.argv) < 4:
    print(f"Usage: python {sys.argv[0]} <env> <gpu-device> <seed>")
    exit()
env = sys.argv[1]
gpu = sys.argv[2]
SEED = int(sys.argv[3])

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_num_threads(1)
device = torch.device(gpu if torch.cuda.is_available() else "cpu")

# env = "PongNoFrameskip-v4"
batch_size = 16
H = W = 84
num_frames = 2
steps = 250000
lr = 1e-3
max_grad_norm = 5.0

wandb.init(project="thesis", entity="epai", tags=["VideoObjectSegmentation-V3"])
wandb.config.update({
        "seed": SEED,
        "env": env,
        "batch-size": batch_size,
        "H": H,
        "W": W,
        "num_frames": num_frames,
        "steps": steps,
        "lr": lr,
        "max_grad_norm": max_grad_norm
    })

data = Dataset(env, batch_size, num_frames, H, W)

model = VideoObjectSegmentationModel(device=device)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
lamb = lambda epoch : 1 - epoch/steps
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lamb)

best_val_loss = 1.
for i in range(steps):
    model.train()
    optimizer.zero_grad()
    inp = data.get_batch("train").to(device)
    x0_ = model(inp)
    x0 = torch.unsqueeze(inp[:, 0, :, :], 1)
    tr_loss = model.compute_loss(x0, x0_)
    tr_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    scheduler.step()
    
    model.eval()
    inp = data.get_batch("val").to(device)
    x0_ = model(inp)
    x0 = torch.unsqueeze(inp[:, 0, :, :], 1)
    val_loss = model.compute_loss(x0, x0_)
    
    model.update_reg()

    if val_loss <= best_val_loss:
        best_val_loss = val_loss
        # print(f"Ep: {i} new best val_loss : {val_loss}")
        # torch.save(model.state_dict(), f"model_{i}.pt")
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, env + '.pt'))

    # if i % 100 == 0:
    #     print(f"Step: {i:5d} - TLoss: {tr_loss.item():.8f} - VLoss: {val_loss.item():.8f} - BestV: {best_val_loss:.8f} - RelErr: {rel_err:.8f}")

    last_lr = scheduler.get_last_lr()[0]
    wandb.log({
            "train_loss": tr_loss,
            "val_loss": val_loss,
            "best_val": best_val_loss,
            "reg_par": model.of_reg_cur,
            "lr": last_lr
        }, step=i)
torch.save(model.state_dict(), os.path.join(wandb.run.dir, env + '_final.pt'))

# import hiddenlayer as hl
# transforms = [ hl.transforms.Prune('Constant') ]
# graph = hl.build_graph(model, IN, transforms=transforms)
# graph.theme = hl.graph.THEMES['blue'].copy()
# graph.save('magic', format='png')