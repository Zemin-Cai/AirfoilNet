import torch
from torch import nn, einsum
from einops import rearrange


class Attention(nn.Module):
    def __init__(
        self,
        *,
        args,
        heads=4,
        dim_head=16,
    ):
        super().__init__()
        self.args = args
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_qk = nn.Linear(dim_head, inner_dim, bias=False)
        self.k = nn.Linear(dim_head, inner_dim, bias=False)

    def forward(self, q, k):
        # fmap: cat([q, k])
        heads, b, f = self.heads, *q.shape

        # q, k = self.to_qk(fmap).chunk(2, dim=-1)
        q = self.to_qk(q)
        k = self.k(k)
        q, k = map(lambda t: rearrange(t, 'b (h d) -> b d h', h=heads), (q, k))
        q = self.scale * q

        sim = einsum('b d c, b e c -> b d e', q, k)

        # sim = rearrange(sim, 'b h x y u v -> b h (x y) (u v)')
        attn = sim.softmax(dim=-1)


        return attn


class Aggregate(nn.Module):
    def __init__(
        self,
        args,
        heads = 4,
        dim_head = 16,
    ):
        super().__init__()
        self.args = args
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_v = nn.Linear(dim_head, inner_dim, bias=False)

        self.gamma = nn.Parameter(torch.zeros(1))

        if dim_head != inner_dim:
            self.project = nn.Linear(inner_dim, dim_head, bias=False)
        else:
            self.project = None

    def forward(self, attn, fmap):
        heads, b, f = self.heads, *fmap.shape

        v = self.to_v(fmap)
        v = rearrange(v, 'b (h d) -> b d h', h=heads)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, 'b d h -> b (h d)', h=heads, d=f)

        if self.project is not None:
            out = self.project(out)

        out = fmap + self.gamma * out

        return out


if __name__ == '__main__':
    f = torch.rand(2, 69, 16)
    model = Attention(args=None)
    att = model(f)
    model = Aggregate(args=None)
    print(model(att, f).shape)