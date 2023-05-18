import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools


class MLP(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim=128):

        super().__init__()

        act = nn.SiLU()

        self.in_dim = input_dim + 1
        self.out_dim = out_dim

        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, self.out_dim),
        )

    def forward(self, u, t=None):
        h = torch.cat([u, t.reshape(-1, 1)], dim=1)
        output = self.net(h)
        return output
