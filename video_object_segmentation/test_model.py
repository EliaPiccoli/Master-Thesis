import torch
from model import VideoObjectSegmentationModel
from dataset import Dataset

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

env = "MsPacmanNoFrameskip-v4"
batch_size = 16
H = W = 84
num_frames = 2
steps = 2 #250000

data = Dataset(env, batch_size, num_frames, H, W)

model = VideoObjectSegmentationModel()
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for i in range(steps):
    model.train()
    inp = data.get_batch("train").to(device)
    out, of = model(inp)
    loss = model.compute_loss(out, of)
    print("Train loss:", loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    inp = data.get_batch("val").to(device)
    out, of = model(inp)
    loss = model.compute_loss(out, of)
    print("Val loss:", loss)

# TODO: setup wandb

# import hiddenlayer as hl
# transforms = [ hl.transforms.Prune('Constant') ]
# graph = hl.build_graph(model, IN, transforms=transforms)
# graph.theme = hl.graph.THEMES['blue'].copy()
# graph.save('magic', format='png')