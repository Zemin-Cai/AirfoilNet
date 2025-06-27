import torch
from torch import nn
import math
from torch.nn import functional as F
from core.condition_encoding import ConditionEncoder

MAX_FREQ = 1000

def get_timestep_embedding(timesteps: torch.Tensor,
                           embedding_dim: int):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(MAX_FREQ) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


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

class FeedForward(nn.Module):
    def __init__(self, dim_in, dim_out=None, dim_mult=1):
        super(FeedForward, self).__init__()
        inner_dim = int(dim_in * dim_mult)
        if dim_out is None:
            dim_out = dim_in

        self.ff = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, dim_out),
        )

    def forward(self, hidden_states):
        return self.ff(hidden_states)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, emb_dim=None, nums_head=4, bias=False, dropout=0.2):
        super(CrossAttention, self).__init__()
        context_dim = context_dim if context_dim is not None else query_dim

        self.q = nn.Linear(query_dim, emb_dim)
        self.k = nn.Linear(context_dim, emb_dim)
        self.v = nn.Linear(context_dim, emb_dim)
        self.att = nn.MultiheadAttention(emb_dim, nums_head, batch_first=True, dropout=dropout)
        self.out = nn.Linear(emb_dim, query_dim)

    def forward(self, x, condition=None):
        condition = condition if condition is not None else x

        q = self.q(x)
        k = self.k(condition)
        v = self.v(condition)

        att, att_weight = self.att(q, k, v)

        out =self.out(att)

        return out

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, emb_dim=None, nums_head=4, bias=False, drop_out=0.):
        super(Attention, self).__init__()

        self.att1 = CrossAttention(query_dim, context_dim=None, emb_dim=emb_dim, nums_head=nums_head, bias=bias, dropout=drop_out)
        self.att2 = CrossAttention(query_dim, context_dim=context_dim, emb_dim=emb_dim, nums_head=nums_head, bias=bias, dropout=drop_out)
        self.ff = FeedForward(query_dim)
        self.ln1 = nn.LayerNorm(query_dim)
        self.ln2 = nn.LayerNorm(query_dim)
        self.ln3 = nn.LayerNorm(query_dim)



    def forward(self, x, condition):

        res = x

        x = self.att1(self.ln1(x))
        x = self.att2(self.ln2(x), condition)
        x = self.ff(self.ln3(x)) + res

        return x

class ResBlock(nn.Module):
    def __init__(self, in_ch, hidden_dim, time_embed, condition_embed, drop_out=0.1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.Linear(in_ch, hidden_dim),
                                   nn.SiLU())

        self.conv2 = nn.Sequential(nn.Linear(hidden_dim, in_ch),
                                   nn.Dropout(drop_out),
                                   nn.SiLU())

        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed, hidden_dim)
        )

        self.condition_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_embed, hidden_dim)
        )

        self.shortcut = nn.Identity()

    def forward(self, x, time, condition):

        h = self.conv1(x)

        c_embed = self.condition_emb(condition)
        t_embed = self.time_emb(time)
        t = t_embed + c_embed
        h += t
        h = self.conv2(h)

        return h + self.shortcut(x)


class ATransformer(nn.Module):
    def __init__(self, in_ch,
                 context_dim=None,
                 time_embed=None,
                 hidden_dim=None,
                 nums_head=4,
                 bias=False,
                 layer_nums=[4],
                 drop_out=0.2,
                 re_embedding_dim=None):
        super(ATransformer, self).__init__()
        self.time_embed = time_embed


        self.cond_f = ConditionEncoder(out_ch=context_dim, re_embedding_dim=re_embedding_dim)


        self.tim_pro = nn.Sequential(nn.Linear(time_embed, 4 * time_embed),
                                     nn.SiLU(),
                                     nn.Linear(4 * time_embed, 4 * time_embed))

        # self.res_blocks = nn.ModuleList([ResBlock(in_ch, hidden_dim, time_embed, time_embed, drop_out=drop_out)
        #                                  for _ in range(layer_nums)])
        #
        # self.att_blocks = nn.ModuleList([Attention(in_ch,
        #                                        context_dim=time_embed,
        #                                        emb_dim=hidden_dim,
        #                                        nums_head=nums_head,
        #                                        bias=bias,
        #                                        drop_out=drop_out)
        #                              for _ in range(layer_nums)])

        self.m = nn.ModuleList([])
        time_embed = 4 * time_embed
        for i in layer_nums:
            for j in range(i):
                self.m.append(ResBlock(in_ch, hidden_dim, time_embed, context_dim, drop_out))

            # self.m.append(Attention(in_ch, time_embed, hidden_dim, nums_head, bias, drop_out))

    def forward(self, x, t, condition=None):
        #condition [geometry, re, alpha]
        t = timestep_embedding(t, self.time_embed)

        condition = self.cond_f(*condition)

        t = self.tim_pro(t)

        # for res, att in zip(self.res_blocks, self.att_blocks):
        #
        #     x = att(x, condition)
        #     x = x.permute(0, 2, 1).contiguous()
        #     x = res(x, t, condition)
        #     x = x.permute(0, 2, 1).contiguous()
        for module in self.m:
            if isinstance(module, ResBlock):

                x = module(x, t, condition)

            # elif isinstance(module, Attention):
            #     x = module(x, condition)

        return x

def count_parameters(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


if __name__ == '__main__':
    """
    input: B, 257, 9 AOA Re Cm Cl Cdt Cdp u v p
    condition: B, 257, 2 x, y
    out:   B, 257, 9 AOA Re Cm Cl Cdt Cdp u v p
    """
    B = 2
    ca = ATransformer(in_ch=2,
                 context_dim=128,
                 time_embed=128,
                 hidden_dim=256,
                 nums_head=4,
                 bias=False,
                 layer_nums=[2, 4, 2],
                      re_embedding_dim=128)

    print(count_parameters(ca))

    x = torch.rand(B, 2)

    t = torch.ones(B)
    condition = [torch.rand(B, 69, 2), torch.rand(B, 1), torch.rand(B, 1)]
    out = ca(x, t, condition)

    print(out.shape)
