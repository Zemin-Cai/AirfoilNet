import torch
from torch import nn
import math
from torch.nn import functional as F
from core.condition_encoding import ConditionEncoder
from abc import abstractmethod


# use sinusoidal position embedding to encode time step (https://arxiv.org/abs/1706.03762)
def timestep_embedding(timesteps, dim, max_period=10000):
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
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        ""

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):

    def forward(self, x, t_emb, c_emb, mask):
        for layer in self:
            if(isinstance(layer, TimestepBlock)):
                x = layer(x, t_emb, c_emb, mask)
            else:
                x = layer(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, in_dim, n_head=1):
        super(MultiHeadAttention, self).__init__()

        self.qkv = nn.Linear(in_dim, 3 * in_dim)
        self.n_head = n_head
        self.proj = nn.Linear(in_dim, in_dim)

    def forward(self, x):
        B, D = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.reshape(B, self.n_head, -1).chunk(3, dim=-1)

        score = (q @ k.transpose(2, 1)) / (math.sqrt(D//self.n_head))
        score = score.softmax(-1)

        h = score @ v
        h = h.reshape(B, D)

        return x + self.proj(h)


class FeedForward(TimestepBlock):
    def __init__(self, in_dim, h_dim, time_dim, condition_dim, dropout=0):
        super(FeedForward, self).__init__()

        self.w1 = nn.Linear(in_dim, h_dim)
        self.w2 = nn.Linear(h_dim, in_dim)

        self.layer_norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, h_dim)
        )

        self.condition_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, h_dim)
        )

    def forward(self, x, timestep, condition, condition_mask):
        res = x

        h = F.silu(self.w1(x))

        timestep = self.time_emb(timestep)
        condition = self.condition_emb(condition) * condition_mask[:, None]


        h += (timestep + condition)
        h = self.w2(h)
        h = self.dropout(h)

        h += res
        h = self.layer_norm(h)

        return h


class EmbedX(FeedForward):
    def __init__(self, in_dim, h_dim, out_dim, time_dim, condition_dim):
        super(EmbedX, self).__init__(in_dim, h_dim, time_dim, condition_dim)
        self.w2 = nn.Linear(h_dim, out_dim)

        self.layer_norm = nn.LayerNorm(out_dim, eps=1e-6)
    def forward(self, x, timestep, condition, condition_mask):


        h = F.silu(self.w1(x))

        timestep = self.time_emb(timestep)
        condition = self.condition_emb(condition) * condition_mask[:, None]

        h += (timestep + condition)
        h = self.w2(h)
        h = self.dropout(h)

        h = F.silu(self.layer_norm(h))

        return h


class TransformerBlock(nn.Module):
    def __init__(self, in_dim, h_dim, time_dim, condition_dim, dropout=0, head=1):
        super(TransformerBlock, self).__init__()

        self.mha = MultiHeadAttention(in_dim, head)
        self.ff = FeedForward(in_dim, h_dim, time_dim, condition_dim, dropout)

    def forward(self, x, timestep, condition, condition_mask):
        x = self.mha(x)
        x = self.ff(x, timestep, condition, condition_mask)

        return x


class ATransformer(nn.Module):
    def __init__(self, in_dim, h_dim, model_dim, condition_dim, dropout=0, head=1, mask=0.1, layer_nums=6, re_embedding_dim=32):
        super(ATransformer, self).__init__()
        time_dim = model_dim * 4
        self.model_dim = model_dim

        # self.x_embedding = FeedForward(in_dim, h_dim, time_dim, condition_dim, dropout=0)
        self.x_embedding = EmbedX(in_dim, h_dim, model_dim, time_dim, condition_dim)

        self.condition_emb = ConditionEncoder(re_embedding_dim=re_embedding_dim, out_ch=condition_dim)



        self.time_emb = nn.Sequential(nn.Linear(model_dim, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim))



        self.block = nn.ModuleList([TransformerBlock(model_dim, h_dim, time_dim, condition_dim, dropout=dropout, head=head)
                                    for _ in range(layer_nums)])

        self.out = nn.Sequential(
            nn.Linear(model_dim, in_dim),

        )

        self.mask = mask

    def forward(self, x, timestep, condition):
        timestep = self.time_emb(timestep_embedding(timestep, self.model_dim))
        condition = self.condition_emb(*condition)

        x = self.x_embedding(x, timestep, condition, condition_mask=torch.ones(x.shape[0], device=x.device))


        z = torch.rand(x.shape[0])
        batch_mask = (z > self.mask).int().to(x.device)

        for m in self.block:
            x = m(x, timestep, condition, batch_mask)

        return self.out(x)

if __name__ == '__main__':
    model = ATransformer(2, 256, 128, 128)
    x = torch.randn(16, 2)
    condition = [torch.randn(16, 69, 2), torch.randn(16, 1), torch.randn(16, 1)]
    timestep = torch.ones(16)

    print(model(x, timestep, condition).shape)




