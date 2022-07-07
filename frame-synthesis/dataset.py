import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

class Dataset(Dataset):
    def __init__(self, path):
        super().__init__()

        self.path = path

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        n, t = index
        # input
        img1 = torch.from_numpy(np.array(Image.open(f"{self.path}/{n}/{t}.png")))
        img2 = torch.from_numpy(np.array(Image.open(f"{self.path}/{n}/{t+1}.png")))
        # output
        img3 = torch.from_numpy(np.array(Image.open(f"{self.path}/{n}/{t+2}.png")))

        return img1, img2, img3

class Sampler(Sampler):
    def __init__(self, idxs, max_len):
        super().__init__()

        self.idxs = idxs
        self.max_ep_len = max_len
    
    def __iter__(self):
        while True:
            n = np.choice(self.idxs)
            t = np.random.randint(0, self.max_ep_len - 3)
            yield n, t

    def __len__(self):
        raise NotImplementedError