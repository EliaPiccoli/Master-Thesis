import numpy as np
import glob
import os
import random
import torch
from PIL import Image

class Dataset():
    def __init__(self, env):
        self.data_path = f'data/{env}/sfmnet/episodes/*'
        # print('data path is {}'.format(self.data_path))

        self.episode_paths = sorted(glob.glob(self.data_path))
        # print('found {} episode paths'.format(len(self.episode_paths)))

        self.episodes = {}
        for episode_path in self.episode_paths:
            self.episodes[episode_path] = sorted(
                glob.glob(os.path.join(episode_path, '*.png')),
                key=lambda x: int(os.path.basename(x).split('_')[0]),
            )
        # print("episodes dict size", len(self.episodes))

        episode_keys = self.episodes.keys()
        self.episodes = {k : self.episodes[k] for k in episode_keys}
        self.valid_keys = [k for k in self.episodes.keys() if len(self.episodes[k]) >= 2]
        self.probs = np.array([len(self.episodes[k]) for k in self.valid_keys], dtype=np.float32)
        self.probs /= np.sum(self.probs)

    def get_batch(self, batch_size, num_frames, H, W):
        frames = np.zeros([batch_size, num_frames, H, W])
        for bs in range(batch_size):
            k = np.random.choice(self.valid_keys, p=self.probs)
            f = self.episodes[k]
            idx = random.randint(0, len(f) - num_frames)
            for j in range(num_frames):
                # print(bs, j)
                path = f[idx + j]
                x = np.expand_dims(np.array(Image.open(path)), axis=0)
                # print(x.shape)
                frames[bs, j:j+1, :, :] = x
        
        return torch.from_numpy(frames).float()