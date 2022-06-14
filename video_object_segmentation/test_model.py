import torch
from model import VideoObjectSegmentationModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = VideoObjectSegmentationModel()
model.to(device)

IN = torch.rand(32, 2, 84, 84, device=device)
print(IN.shape)

out = model(IN)
print("O:", out.shape)

# import hiddenlayer as hl
# transforms = [ hl.transforms.Prune('Constant') ]
# graph = hl.build_graph(model, IN, transforms=transforms)
# graph.theme = hl.graph.THEMES['blue'].copy()
# graph.save('magic', format='png')

# TODO
# 1- Create dataset
# 2- Training cycle