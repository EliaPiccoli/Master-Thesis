from argparse import Namespace
import sys
sys.path.append('../')

import torch
import numpy as np

from model import PNN, PNNCol
from atariari.methods.encoders import NatureCNN
from video_object_segmentation.model import VideoObjectSegmentationModel
from keypoints_transporter.models import Encoder, KeyNet, RefineNet, Transporter

device = 'cpu' # torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

state_path = "/data/e.piccoli/Master-Thesis/state_representation/wandb/run-20220518_164211-3ew92jnk/files/encoder.pt"
n = Namespace()
setattr(n, 'feature_size', 256)
setattr(n, 'no_downsample', True)
setattr(n, 'end_with_relu', False)
state_rep_encoder = NatureCNN(1, n)
state_rep_encoder.load_state_dict(torch.load(state_path, map_location=device))

video_path = "/data/e.piccoli/Master-Thesis/video_object_segmentation/wandb/run-20220707_191800-124hx1ez/files/PongNoFrameskip-v4.pt"
video_segmenation_model = VideoObjectSegmentationModel(device=device, K=20)
video_segmenation_model.load_state_dict(torch.load(video_path, map_location=device))

keypoints_path = "/data/e.piccoli/Master-Thesis/keypoints_transporter/wandb/run-20220623_163543-34u2d8d0/files/PongNoFrameskip-v4.pt"
e = Encoder(3)
k = KeyNet(3, 4)
r = RefineNet(3)
keypoints_model = Transporter(e, k, r)
keypoints_model.load_state_dict(torch.load(keypoints_path, map_location=device))

skills = [
    ("state-representation", state_rep_encoder),
    ("video-segmentation", video_segmenation_model),
    ("keypoints", keypoints_model)
]

col1 = PNNCol(0, 154, 5, (154, 16, 16), skills)
col1.to(device)
col1.train()

pnn = PNN([col1])
pnn.to(device)
pnn.train()

inp = torch.rand(1, 3, 84, 84, device=device)
out = pnn(inp)
print("PNN_OUT:", out.shape)