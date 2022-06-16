import torch
from model import VideoObjectSegmentationModel
from kornia.losses import ssim_loss
from dataset import Dataset

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

env = "MsPacmanNoFrameskip-v4"
batch_size = 16
H = W = 84
num_frames = 2

data = Dataset(env)

model = VideoObjectSegmentationModel()
model.to(device)

for i in range(2):
    # print("Step:", i)
    inp = data.get_batch(batch_size, num_frames, H, W).to(device)
    # print("IN:", inp.shape)
    # print(inp)

    out = model(inp)
    # print("OUT:", out)

    x = torch.unsqueeze(out[:, 0, :, :], 1)
    y = torch.unsqueeze(out[:, 1, :, :], 1)

    loss = (1 - ssim_loss(x, y, 11))/2
    # print("loss:", loss, loss.shape)

# TODO: define real training cycle and train model (setup wandb)

# import hiddenlayer as hl
# transforms = [ hl.transforms.Prune('Constant') ]
# graph = hl.build_graph(model, IN, transforms=transforms)
# graph.theme = hl.graph.THEMES['blue'].copy()
# graph.save('magic', format='png')