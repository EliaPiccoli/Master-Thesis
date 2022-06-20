import wandb
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from models import Encoder, KeyNet, RefineNet, Transporter, transport
from dataset import Dataset, Sampler

torch.set_num_threads(1)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

ENV = "MsPacmanNoFrameskip-v4"
data_path = f"data/{ENV}"
NUM_EPS = 100
MAX_EP_LEN = 100
MAX_ITER = 1e6
batch_size = 64
image_channels = 3
K = 4
lr = 1e-3
lr_decay = 0.95
lr_deacy_len = 1e5

wandb.init(project="thesis", entity="epai", tags=["ObjectKeyPoints"])
wandb.config.update({
        "env": ENV,
        "num_eps": NUM_EPS,
        "max_ep_len": MAX_EP_LEN,
        "steps": MAX_ITER,
        "batch-size": batch_size,
        "image-channels": image_channels,
        "K": K,
        "lr": lr,
        "lr-decay": lr_decay,
        "lr-decay-len": lr_deacy_len
    })

encoder = Encoder(image_channels)
key_net = KeyNet(image_channels, K)
refine_net = RefineNet(image_channels)
transporter = Transporter(encoder, key_net, refine_net)
transporter.to(device)
transporter.train()

dataset = Dataset(data_path, NUM_EPS, MAX_EP_LEN, transforms.ToTensor())
sampler = Sampler(dataset)
data_loader = DataLoader(dataset, batch_size, sampler=sampler, num_workers=4)

optimizer = torch.optim.Adam(transporter.parameters(), lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_deacy_len, lr_decay)

for i, (xt, xtp1) in enumerate(data_loader):
    if i > MAX_ITER:
        break
    xt = xt.to(device)
    xtp1 = xtp1.to(device)
    optimizer.zero_grad()
    reconstruction = transporter(xt, xtp1)
    loss = F.mse_loss(reconstruction, xtp1)
    loss.backward
    optimizer.step()
    scheduler.step()

    wandb.log({
            "loss": loss,
            "lr": scheduler.get_last_lr()[0]
        }, step=i)

    if i % 100 == 0:
        print(f"Step: {i} - Loss: {loss.item()}")
        torch.save(transporter.state_dict(), os.path.join(wandb.run.dir, ENV + '.pt'))