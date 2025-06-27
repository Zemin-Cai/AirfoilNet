from torch import nn
import torch
from core.feature_extractor import FeatureExtractor
from core.vector_embedding import ConditionEmbedding, GeometryEmbedding
import math


def re_embedding(timesteps, dim, max_period=5):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """


    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps.float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class ConditionEncoder(nn.Module):
    def __init__(self, geometry_dim=69, re_embedding_dim=32,
                 out_ch=2):
        super(ConditionEncoder, self).__init__()

        self.re_embedding_dim = re_embedding_dim

        self.re_embeeding_ = nn.Sequential(
            nn.Linear(self.re_embedding_dim, 64),

            nn.Linear(64, 1)
        )

        self.con_embedding = ConditionEmbedding(2)
        self.gemetry_embedding = GeometryEmbedding()

        self.mlp_head = nn.Sequential(
            nn.Linear(geometry_dim * 2, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, geometry_dim * 2),
            nn.LeakyReLU()
        )

        self.head = nn.ModuleList([])
        [self.head.append(self.mlp_head) for _ in range(2)]

        self.out = nn.Sequential(
            nn.Linear(geometry_dim * 2, 64),

            nn.Linear(64, 64),

            nn.Linear(64, out_ch)
        )

    def forward(self, geometry, re, alpha):
        re = re_embedding(re, dim=self.re_embedding_dim)

        re = self.re_embeeding_(re)
        condition = torch.cat([re, alpha], dim=-1)
        condition = self.con_embedding(condition)
        # geometry = geometry.reshape(geometry.shape[0], -1)
        # geometry = self.gemetry_embedding(geometry)

        geometry = geometry + condition[:, None]

        geometry = self.gemetry_embedding(geometry)
        geometry = condition[:, None] + geometry

        geometry_mlp = geometry.reshape(geometry.shape[0], -1)
        res = geometry_mlp

        for m in self.head:
            geometry_mlp = m(geometry_mlp)
            geometry_mlp = res + geometry_mlp
            res = geometry_mlp

        condition = self.out(geometry_mlp)

        return condition

if __name__ == '__main__':
    x = torch.rand(1, 69, 2)
    re = torch.rand(1, 1)
    a = torch.rand(1, 1)
    model = ConditionEncoder()
    print(model(x, re, a).shape)



