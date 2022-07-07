import numpy as np
SEED = 10
np.random.seed(SEED)

import os
from PIL import Image
from baselines.common.atari_wrappers import make_atari, WarpFrame

ENV = "PongNoFrameskip-v4"
NUM_EPS = 100
IMG_SZ = 256

env = WarpFrame(make_atari(ENV, max_episode_steps=100), width=IMG_SZ, height=IMG_SZ, grayscale=False)
obs = env.reset()
datadir = f"data/{ENV}_{IMG_SZ}"

for ep in range(NUM_EPS):
    os.makedirs(f"{datadir}/{ep}", exist_ok=True)
    obs = env.reset()
    timestep = 0
    img = Image.fromarray(obs)
    img.save(f"{datadir}/{ep}/{timestep}.png")
    
    while True:
        obs, r, done, _ = env.step(env.action_space.sample())
        timestep += 1
        img = Image.fromarray(obs)
        img.save(f"{datadir}/{ep}/{timestep}.png")
        if done:
            break
print("GenerateData complete - saved in:", datadir)