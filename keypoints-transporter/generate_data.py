import numpy as np
SEED = 10
np.random.seed(SEED)

import os
from PIL import Image
from baselines.common.atari_wrappers import make_atari, WarpFrame

ENV = "MsPacmanNoFrameskip-v4"
NUM_EPS = 100

env = WarpFrame(make_atari(ENV, max_episode_steps=100), grayscale=False)
obs = env.reset()
datadir = f"data/{ENV}"

for ep in range(NUM_EPS):
    print("Episode:", ep)
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