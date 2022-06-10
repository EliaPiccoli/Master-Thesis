import torch
from model import VideoObjectSegmentationModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = VideoObjectSegmentationModel()
model.to(device)

IN = torch.rand(32, 2, 84, 84, device=device)
print(IN.shape)

out = model(IN)
print("O:", out.shape)