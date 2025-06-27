import torch
from torch import nn


class ConditionEmbedding(nn.Module):
    def __init__(self, in_dim=16+2):
        super(ConditionEmbedding, self).__init__()

        self.embedding = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 2),
            nn.SiLU()
        )

    def forward(self, condition):

        return self.embedding(condition)

class GeometryEmbedding(nn.Module):
    def __init__(self):
        super(GeometryEmbedding, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(2, 32),
            nn.SiLU(),
            nn.Linear(32, 2),
            nn.SiLU()
        )

    def forward(self, x):
        return self.embedding(x)