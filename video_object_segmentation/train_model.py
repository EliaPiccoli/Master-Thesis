from numpy import Inf
import torch
import wandb
import os
import torch.nn.functional as F
from model import VideoObjectSegmentationModel
from dataset import Dataset

torch.set_num_threads(1)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

env = "PongNoFrameskip-v4"
batch_size = 32
H = W = 84
num_frames = 2
steps = 250000
lr = 1e-4

wandb.init(project="thesis", entity="epai", tags=["VideoObjectSegmentation"])
wandb.config.update({
        "env": env,
        "batch-size": batch_size,
        "H": H,
        "W": W,
        "num_frames": num_frames,
        "steps": steps,
        "lr": lr
    })

data = Dataset(env, batch_size, num_frames, H, W)

model = VideoObjectSegmentationModel()
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

best_val_loss = Inf
for i in range(steps):
    model.train()
    optimizer.zero_grad()
    inp = data.get_batch("train").to(device)
    flow_out, of = model(inp)
    flow_out = torch.reshape(flow_out, (-1, flow_out.size(2), flow_out.size(3), flow_out.size(1)))
    x0 = torch.unsqueeze(inp[:, 0, :, :], 1)
    x0_ = F.grid_sample(x0, flow_out, align_corners=False)
    x1 = torch.unsqueeze(inp[:, 1, :, :], 1)
    tr_loss = model.compute_loss(x0_, x1, of)
    tr_loss.backward()
    optimizer.step()

    model.eval()
    inp = data.get_batch("val").to(device)
    flow_out, of = model(inp)
    flow_out = torch.reshape(flow_out, (-1, flow_out.size(2), flow_out.size(3), flow_out.size(1)))
    x0 = torch.unsqueeze(inp[:, 0, :, :], 1)
    x0_ = F.grid_sample(x0, flow_out, align_corners=False)
    x1 = torch.unsqueeze(inp[:, 1, :, :], 1)
    val_loss = model.compute_loss(x0_, x1, of)

    if val_loss < best_val_loss:
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, env + '.pt'))
        best_val_loss = val_loss

    if i % 100 == 0:
        print(f"Step: {i} - TLoss: {tr_loss.item()} - VLoss: {val_loss.item()}")
    wandb.log({
            "train_loss": tr_loss,
            "val_loss": val_loss,
            "reg_par": model.of_reg_cur
        }, step=i)

# torch.save(model.state_dict(), os.path.join(wandb.run.dir, env + '.pt'))

# import hiddenlayer as hl
# transforms = [ hl.transforms.Prune('Constant') ]
# graph = hl.build_graph(model, IN, transforms=transforms)
# graph.theme = hl.graph.THEMES['blue'].copy()
# graph.save('magic', format='png')