import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

class Dataset(Dataset):
    def __init__(self, path, idxs, max_len):
        super().__init__()

        self.path = path
        self.input_mean = [0.5 * 255, 0.5 * 255, 0.5 * 255]
        self.input_std = [0.5 * 255, 0.5 * 255, 0.5 * 255]
        self.idxs = idxs
        self.max_ep_len = max_len

    def __len__(self):
        return len(self.idxs)

    def _normalize(self, img, mean, std):
        img = img - np.array(mean)[np.newaxis, np.newaxis, ...]
        img = img / np.array(std)[np.newaxis, np.newaxis, ...]
        return img

    def __getitem__(self, idx):
        n = idx
        t = np.random.randint(0, self.max_ep_len - 3)
        imgs = []
        imgs.append(np.array(Image.open(f"{self.path}/{n}/{t}.png")))
        imgs.append(np.array(Image.open(f"{self.path}/{n}/{t+1}.png")))
        imgs.append(np.array(Image.open(f"{self.path}/{n}/{t+2}.png")))

        for i in range(len(imgs)):
            imgs[i] = self._normalize(imgs[i], self.input_mean, self.input_std)
            imgs[i] = torch.from_numpy(imgs[i]).permute(2, 0, 1).contiguous().float()

        return torch.cat([imgs[0], imgs[1]], 0), imgs[-1]