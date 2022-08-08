import sys
sys.path.append('../')

import torch
import wandb
import random
import numpy as np

from baselines.common.atari_wrappers import make_atari, WarpFrame
from argparse import Namespace
from model import PNN, PNNCol
from agent import Agent
from atariari.methods.encoders import NatureCNN
from video_object_segmentation.model import VideoObjectSegmentationModel
from keypoints_transporter.models import Encoder, KeyNet, RefineNet, Transporter

device = 'cpu'
torch.set_num_threads(8)

BATCH_SIZE = 64
MEMORY_SIZE = 50000
GAMMA = 0.97
TAU = 0.05
LR = 5e-4
EPS = 1.0
EPS_MIN = 0.02
EPS_DECAY = 0.995
EPISODES = 100000
MAX_STEP = 10000
SAVE_CKPT = 1000
GRAD_CLIP = 40
ENV1 = "PongNoFrameskip-v4"
ENV2 = "MsPacmanNoFrameskip-v4"
SEED = 0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

### Load model previous game (PONG)
# create env
env = WarpFrame(make_atari(ENV1), width=84, height=84, grayscale=False)
env.seed(SEED)
ACTION_SPACE = env.action_space.n

# load skills models
state_path = "/data/e.piccoli/Master-Thesis/state_representation/wandb/run-20220722_190610-37a6tv3d/files/PongNoFrameskip-v4.pt"
n = Namespace()
setattr(n, 'feature_size', 256)
setattr(n, 'no_downsample', True)
setattr(n, 'end_with_relu', False)
state_rep_encoder = NatureCNN(1, n)
state_rep_encoder.load_state_dict(torch.load(state_path, map_location=device))
state_rep_encoder.eval()

video_path = "/data/e.piccoli/Master-Thesis/video_object_segmentation/wandb/run-20220707_191800-124hx1ez/files/PongNoFrameskip-v4.pt"
video_segmenation_model = VideoObjectSegmentationModel(device=device, K=20)
video_segmenation_model.load_state_dict(torch.load(video_path, map_location=device))
video_segmenation_model.eval()

keypoints_path = "/data/e.piccoli/Master-Thesis/keypoints_transporter/wandb/run-20220623_163543-34u2d8d0/files/PongNoFrameskip-v4.pt"
e = Encoder(3)
k = KeyNet(3, 4)
r = RefineNet(3)
keypoints_model = Transporter(e, k, r)
keypoints_model.load_state_dict(torch.load(keypoints_path, map_location=device))
keypoints_model.eval()

skills = [
    ("state-representation", state_rep_encoder),
    ("video-segmentation", video_segmenation_model),
    ("keypoints", keypoints_model)
]

col1 = PNNCol(0, ACTION_SPACE, skills, 154, (154, 16, 16))
pnn = PNN([col1])
pnn.eval()
model_path = "/data/e.piccoli/Master-Thesis/progressive_nn/wandb/run-20220727_144002-39ii4aus/files/model_40000.pt" #sqdn1
# model_path = "/data/e.piccoli/Master-Thesis/progressive_nn/wandb/run-20220802_021225-1ui6exeh/files/model_36000.pt" # sqdn3
pnn.load_state_dict(torch.load(model_path, map_location=device))
pong_col = pnn.columns[0]

### Create model new game (PACMAN)
# create env
env2 = WarpFrame(make_atari(ENV2), width=84, height=84, grayscale=False)
env2.seed(SEED)
ACTION_SPACE2 = env2.action_space.n

# load skills models
state_path2 = "/data/e.piccoli/Master-Thesis/state_representation/wandb/run-20220518_164211-3ew92jnk/files/MsPacmanNoFrameskip-v4.pt"
n = Namespace()
setattr(n, 'feature_size', 256)
setattr(n, 'no_downsample', True)
setattr(n, 'end_with_relu', False)
state_rep_encoder2 = NatureCNN(1, n)
state_rep_encoder2.load_state_dict(torch.load(state_path2, map_location=device))
state_rep_encoder2.eval()

video_path2 = "/data/e.piccoli/Master-Thesis/video_object_segmentation/wandb/run-20220808_102320-2ghdq07i/files/MsPacmanNoFrameskip-v4.pt"
video_segmenation_model2 = VideoObjectSegmentationModel(device=device, K=20)
video_segmenation_model2.load_state_dict(torch.load(video_path2, map_location=device))
video_segmenation_model2.eval()

keypoints_path2 = "/data/e.piccoli/Master-Thesis/keypoints_transporter/wandb/run-20220622_111625-2ke0zp2g/files/MsPacmanNoFrameskip-v4.pt"
e = Encoder(3)
k = KeyNet(3, 4)
r = RefineNet(3)
keypoints_model2 = Transporter(e, k, r)
keypoints_model2.load_state_dict(torch.load(keypoints_path2, map_location=device))
keypoints_model2.eval()

skills2 = [
    ("state-representation", state_rep_encoder2),
    ("video-segmentation", video_segmenation_model2),
    ("keypoints", keypoints_model2)
]

# create models
col2 = PNNCol(1, ACTION_SPACE2, skills2, 154, (154, 16, 16))
col2.to(device)
col2.train()

pnn2 = PNN([pong_col, col2])
pnn2.to(device)
pnn2.train()

# target network
t_col2 = PNNCol(1, ACTION_SPACE2, skills2, 154, (154, 16, 16))
t_col2.to(device)
t_col2.train()

t_pnn2 = PNN([pong_col, t_col2])
t_pnn2.to(device)
t_pnn2.train()

# create agent
args = {
    'batch_size': BATCH_SIZE,
    'gamma': GAMMA,
    'tau': TAU,
    'lr': LR,
    'eps': EPS,
    'eps_decay': EPS_DECAY,
    'eps_min': EPS_MIN,
    'max_episodes': EPISODES,
    'max_ep_step': MAX_STEP,
    'save_ckpt': SAVE_CKPT,
    'memory_size': MEMORY_SIZE,
    'grad_clip': GRAD_CLIP
}

agent = Agent(env2, args, pnn2, t_pnn2, wandb, None)
# agent.train()

state = env2.reset()
with torch.no_grad():
    s = torch.from_numpy(state).float().unsqueeze(0).to(device)
    action = np.argmax(pnn2(s).cpu())
    print(action)