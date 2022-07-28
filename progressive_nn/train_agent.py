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

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(8)

BATCH_SIZE = 64
MEMORY_SIZE = 50000
GAMMA = 0.97
TAU = 0.05
LR = 5e-4
EPS = 1.0
EPS_MIN = 0.05
EPS_DECAY = 0.995
EPISODES = 100000
MAX_STEP = 10000
SAVE_CKPT = 1000
GRAD_CLIP = 40
ENV = "PongNoFrameskip-v4"
SEED = 1

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

wandb.init(project="thesis", entity="epai", tags=["Pong-SDQN"])

# create env
env = WarpFrame(make_atari(ENV), width=84, height=84, grayscale=False)
ACTION_SPACE = env.action_space.n

# create skills models
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

# update all infos
wandb.config.update({
        'seed': SEED,
        'env': ENV,
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
        'action_space': ACTION_SPACE,
        'grad_clip': GRAD_CLIP,
        'state_model': state_path,
        'video_model': video_path,
        'key_model': keypoints_path
    })

# create models
col1 = PNNCol(0, ACTION_SPACE, skills, 154, (154, 16, 16))
col1.to(device)
col1.train()

pnn = PNN([col1])
pnn.to(device)
pnn.train()

# target network
t_col1 = PNNCol(0, ACTION_SPACE, skills, 154, (154, 16, 16))
t_col1.to(device)
t_col1.train()

t_pnn = PNN([t_col1])
t_pnn.to(device)
t_pnn.train()

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

agent = Agent(env, args, pnn, t_pnn, wandb, device)
agent.train()