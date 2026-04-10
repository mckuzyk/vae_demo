import numpy as np
import torch
from torch.utils.data import Dataset


rng = np.random.default_rng(42)


class Spiral:
    def __init__(self, omega=2 * np.pi, rdot=1, spread=0.01):
        self.omega = omega
        self.rdot = rdot
        self.spread = spread

    def __call__(self, t):
        t = np.atleast_1d(np.asarray(t))
        x, y = self.exact(t)
        x += rng.normal(scale=self.spread, size=t.shape)
        y += rng.normal(scale=self.spread, size=t.shape)
        return x, y

    def exact(self, t):
        r = self.rdot * t
        theta = self.omega * t
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y


class SpiralDataset(Dataset):
    def __init__(self, ts, omega=2 * np.pi, rdot=1, spread=0.01):
        s = Spiral(omega, rdot, spread)
        x, y = s(ts)
        self.omega = omega
        self.rdot = rdot
        self.spread = spread
        self.ts = torch.from_numpy(ts).float().reshape(-1, 1)
        self.x = torch.from_numpy(x).float().reshape(-1, 1)
        self.y = torch.from_numpy(y).float().reshape(-1, 1)
        self.data = torch.cat([self.x, self.y], dim=1)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.ts.shape[0]
