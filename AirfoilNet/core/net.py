from torch import nn
import torch
from core.feature_extractor import FeatureExtractor
from core.vector_embedding import ConditionEmbedding, GeometryEmbedding
from core.am import Aggregate, Attention
import math


def norm_re(x, min=50000, max=1000000, avr=420709.0736040609):
    return (x - avr) / (max - min)

def re_embedding(timesteps, dim, max_period=5):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if dim == 1:
        embedding = norm_re(timesteps)
        return embedding

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps.float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class AGRUNet(nn.Module):
    def __init__(self, cnn_in_ch=1, cnn_out_ch=2, spp_layers=4, geometry_dim=69, gru_layers=12, re_embedding_dim=32,
                 out_ch=2):
        super(AGRUNet, self).__init__()
        """
        geometry_dim denotes the number of coordinate points
        """
        self.feature_encoder = FeatureExtractor(cnn_in_ch, cnn_out_ch, num_layers=spp_layers,
                                                embedding_dim=geometry_dim)

        self.condtion_embedding = ConditionEmbedding(cnn_out_ch + 2)
        self.gemetry_embedding = GeometryEmbedding()

        self.gru_layers = gru_layers

        self.update = UpdateBlock(2, 16)

        self.mlp_head = nn.Sequential(
            nn.Linear(geometry_dim * 2, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100, geometry_dim * 2),
            nn.LeakyReLU()
        )

        self.head = nn.ModuleList([])
        [self.head.append(self.mlp_head) for _ in range(2)]


        self.out = nn.Sequential(
            nn.Linear(geometry_dim * 2, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100, out_ch)
        )

        self.re_embedding_dim = re_embedding_dim

        self.re_embeeding_ = nn.Sequential(
            nn.Linear(self.re_embedding_dim, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 1)
        )

        self.embedding = nn.Sequential(
            nn.Linear(geometry_dim * 2, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 32)
        )

        self.ac = nn.GELU()

    def forward(self, image, geometry, re, alpha, test_mode=False):
        image = 2 * image - 1

        geometry_feature, feature_embedding = self.feature_encoder(image)
        #
        f = feature_embedding.detach()

        re = re_embedding(re, dim=self.re_embedding_dim)
        re = self.re_embeeding_(re)
        condition = torch.cat([re, alpha], dim=-1)

        condition = self.condtion_embedding(f, condition)

        geometry = geometry + condition[:, None]

        geometry = self.gemetry_embedding(geometry)
        geometry = condition[:, None] + geometry

        geometry_mlp = geometry.reshape(geometry.shape[0], -1)
        res = geometry_mlp

        for m in self.head:
            geometry_mlp = m(geometry_mlp)
            geometry_mlp = res + geometry_mlp
            res = geometry_mlp

        out_put_mlp = self.out(geometry_mlp)


        f = self.embedding(geometry.reshape(geometry.shape[0], -1))
        net, inp = torch.split(f, [16, 16], dim=-1)
        net = torch.tanh(net)
        inp = self.ac(inp)
        # att = self.att(inp)

        c = out_put_mlp

        output_list = []

        for i in range(self.gru_layers):
            c = c.detach()

            net, delta_c = self.update(net, inp, c)

            c = c + delta_c

            output_list.append(c)

        if test_mode:
            return geometry_feature, output_list[-1]

        return geometry_feature, output_list, out_put_mlp

class GRU(nn.Module):
    def __init__(self, in_dim=16 + 16, hidden_dim=64):
        super(GRU, self).__init__()

        self.wz = nn.Linear(in_dim + hidden_dim, hidden_dim)
        self.wr = nn.Linear(in_dim + hidden_dim, hidden_dim)
        self.wq = nn.Linear(in_dim + hidden_dim, hidden_dim)

    def forward(self, h, x):
        # h-net-tanh, x-inp-relu
        hx = torch.cat([h, x], dim=-1)

        z = torch.sigmoid(self.wz(hx))
        r = torch.sigmoid(self.wr(hx))
        q = torch.tanh(self.wq(torch.cat([r * h, x], dim=-1)))
        h = (1 - z) * h + z * q

        return h

class UpdateBlock(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=16):
        super(UpdateBlock, self).__init__()
        self.c_encoder = nn.Sequential(nn.Linear(in_dim, hidden_dim - in_dim),
                                       nn.GELU())

        self.gru = GRU(in_dim=hidden_dim + 16, hidden_dim=hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 100, bias=False),
            nn.LeakyReLU(),
            nn.Linear(100, 100, bias=False),
            nn.LeakyReLU(),
            nn.Linear(100, in_dim, bias=False)
        )


        # self.aggregator = Aggregate(args=None, dim_head=16, heads=4)

    def forward(self, net, inp, c):
        c_feature = self.c_encoder(c)
        c_feature = torch.cat([c_feature, c], dim=-1)

        # c_feature_aggre = self.aggregator(attention, c_feature)

        inp = torch.cat([inp, c_feature], dim=-1)

        net = self.gru(net, inp)

        delta_c = self.head(net)

        return net, delta_c

class UpdateBlock_(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=16):
        super(UpdateBlock_, self).__init__()
        self.c_encoder = nn.Sequential(nn.Linear(in_dim, hidden_dim - in_dim),
                                       nn.GELU())

        self.gru = GRU(in_dim=hidden_dim + 16, hidden_dim=hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, in_dim),
        )
        # self.aggregator = Aggregate(args=None, dim_head=16, heads=4)

    def forward(self, net, inp, c):
        c_feature = self.c_encoder(c)
        c_feature = torch.cat([c_feature, c], dim=-1)

        # c_feature_aggre = self.aggregator(attention, c_feature)

        inp = torch.cat([inp, c_feature], dim=-1)

        net = self.gru(net, inp)

        delta_c = self.head(net)

        return net, delta_c

class GRUNet(nn.Module):
    def __init__(self, cnn_in_ch=1, cnn_out_ch=16, spp_layers=4, geometry_dim=69, gru_layers=12, re_embedding_dim=128, out_ch=2):
        super(GRUNet, self).__init__()
        """
        geometry_dim denotes the number of coordinate points
        """
        self.feature_encoder = FeatureExtractor(cnn_in_ch, cnn_out_ch, num_layers=spp_layers,
                                                embedding_dim=geometry_dim)

        self.condtion_embedding = ConditionEmbedding(cnn_out_ch + 2)
        self.gemetry_embedding = GeometryEmbedding()

        self.gru_layers = gru_layers

        self.update = UpdateBlock_(2, 16)

        self.out = nn.Sequential(
            nn.Linear(2, 100),
            nn.LeakyReLU(),
            nn.Linear(100, out_ch)
        )

        self.re_embedding_dim = re_embedding_dim

        self.re_embeeding_ = nn.Sequential(
            nn.Linear(self.re_embedding_dim, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 1)
        )

        self.embedding = nn.Sequential(
            nn.Linear(geometry_dim * 2, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 32)
        )

        self.ac = nn.GELU()

        # self.att = Attention(args=None)

    def forward(self, image, geometry, re, alpha, test_mode=False):
        image = 2 * image - 1

        geometry_feature, feature_embedding = self.feature_encoder(image)
        #
        f = feature_embedding.detach()

        re = re_embedding(re, dim=self.re_embedding_dim)
        re = self.re_embeeding_(re)
        condition = torch.cat([re, alpha], dim=-1)

        condition = self.condtion_embedding(f, condition)

        geometry = geometry + condition[:, None]
        geometry = self.gemetry_embedding(geometry)
        geometry = condition[:, None] + geometry
        geometry = geometry.reshape(geometry.shape[0], -1)

        f = self.embedding(geometry)
        net, inp = torch.split(f, [16, 16], dim=-1)
        net = torch.tanh(net)
        inp = self.ac(inp)
        # att = self.att(inp)

        c = torch.zeros((geometry.shape[0], 2), device=f.device)

        output_list = []

        for i in range(self.gru_layers):
            c = c.detach()

            net, delta_c = self.update(net, inp, c)

            c = c + delta_c

            c = self.out(c)

            output_list.append(c)

        if test_mode:
            return geometry_feature, output_list[-1]

        return geometry_feature, output_list

class MLPNet(nn.Module):
    def __init__(self, cnn_in_ch=1, cnn_out_ch=16, spp_layers=4, geometry_dim=69, re_embedding_dim=128, out_ch=2):
        super(MLPNet, self).__init__()
        """
        geometry_dim denotes the number of coordinate points
        """
        self.feature_encoder = FeatureExtractor(cnn_in_ch, cnn_out_ch, num_layers=spp_layers,
                                                embedding_dim=geometry_dim)

        self.condtion_embedding = ConditionEmbedding(cnn_out_ch + 2)
        self.gemetry_embedding = GeometryEmbedding()

        self.predition_head = nn.Sequential(
            nn.Linear(geometry_dim * 2, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100, geometry_dim * 2),
            nn.LeakyReLU()
        )

        self.head = nn.ModuleList([self.predition_head for _ in range(2)])
        # [self.head.append(self.predition_head) for _ in range(6)]

        self.out = nn.Sequential(
            nn.Linear(geometry_dim * 2, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100, out_ch)
        )

        self.re_embedding_dim = re_embedding_dim

        self.re_embeeding_ = nn.Sequential(
            nn.Linear(self.re_embedding_dim, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 1)
        )


    def forward(self, image, geometry, re, alpha, test_mode=False):
        geometry_feature, feature_embedding = self.feature_encoder(image)

        f = feature_embedding.detach()
        re = re_embedding(re, dim=self.re_embedding_dim)
        re = self.re_embeeding_(re)
        condition = torch.cat([re, alpha], dim=-1)

        condition = self.condtion_embedding(f, condition)

        geometry = geometry + condition[:, None]
        geometry = self.gemetry_embedding(geometry)
        geometry = condition[:, None] + geometry

        geometry = geometry.reshape(geometry.shape[0], -1)
        res = geometry

        for m in self.head:

            geometry = m(geometry)
            geometry = res + geometry
            res = geometry

        out_put = self.out(geometry)

        return geometry_feature, out_put

if __name__ == '__main__':
    x = torch.rand(1, 69, 2)
    image = torch.rand(1, 1, 320, 240)
    re = torch.rand(1, 1)
    aoa = torch.rand(1, 1)
    model = AGRUNet(geometry_dim=69)
    print(model(image, x, re, aoa)[1][1].shape)
