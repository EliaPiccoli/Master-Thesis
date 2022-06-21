import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Sampler

class Dataset(Dataset):
    def __init__(self, path, ep=10, max_ep_len=100, transform=None):
        super().__init__()

        self.path = path
        self.transform = transform
        self.ep = ep
        self.max_ep_len = max_ep_len

    def __len__():
        raise NotImplementedError

    def __getitem__(self, index):
        n, t, tp1 = index
        imgt = np.array(Image.open(f"{self.path}/{n}/{t}.png"))
        imgtp1 = np.array(Image.open(f"{self.path}/{n}/{tp1}.png"))
        if self.transform is not None:
            imgt = self.transform(imgt)
            imgtp1 = self.transform(imgtp1)

        return imgt, imgtp1

    def get_trajectory(self, idx):
        images = [np.array(Image.open('{}/{}/{}.png'.format(self.path, idx, t))) for t in range(self.max_ep_len)]
        return [self.transform(im) for im in images]

class Sampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        while True:
            n = np.random.randint(self.dataset.ep)
            num_images = self.dataset.max_ep_len
            t_ind = np.random.randint(0, num_images - 20)
            tp1_ind = t_ind + np.random.randint(20)
            yield n, t_ind, tp1_ind

    def __len__(self):
        raise NotImplementedError