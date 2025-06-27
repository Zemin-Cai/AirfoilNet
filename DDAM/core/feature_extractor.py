import torch
from torch import nn
import math
from torch.nn import functional as F

class SPPNet(nn.Module):
    def __init__(self, num_layers, pool_type='avg'):
        super(SPPNet, self).__init__()
        self.num_layers = num_layers
        self.pool_type = pool_type

    def forward(self, x):
        b, c, h, w = x.shape

        for i in range(self.num_layers):
            level = i + 1
            kernel_size = (math.ceil(h/level), math.ceil(w/level))
            strid = kernel_size
            padding = (math.floor((kernel_size[0] * level-h+1)/2), math.floor((kernel_size[1] * level-w+1)/2))
            if self.pool_type == 'max':
                tensor = F.max_pool2d(x, kernel_size=kernel_size, padding=padding, stride=strid)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size, padding=padding, stride=strid)

            if i == 0:
                x_all = tensor.reshape(b, -1)
            else:
                x_all = torch.cat([x_all, tensor.reshape(b, -1)], dim=-1)

        return x_all

class FeatureExtractor(nn.Module):
    def __init__(self, in_ch=3, out=16, num_layers=4, embedding_dim=133):
        super(FeatureExtractor, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 10, kernel_size=7, padding=3),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(10, 10, kernel_size=7, padding=3),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(10, 10, kernel_size=7, padding=3),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(10, 20, kernel_size=7, padding=3),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 10, kernel_size=7, padding=3),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
        )

        self.spp_out_dim = sum([(i+1)**2 for i in range(num_layers)]) * 10

        self.spp = SPPNet(num_layers=num_layers)

        self.mlp_1 = nn.Sequential(
            nn.Linear(self.spp_out_dim, 16),
            nn.GELU(),
            nn.Linear(16, 16),
            nn.GELU(),
            nn.Linear(16, out)
        )

        self.mlp_2 = nn.Sequential(
            nn.Linear(out, 16),
            nn.GELU(),
            nn.Linear(16, 16),
            nn.GELU(),
            nn.Linear(16, embedding_dim)
        )

        self.mlp_3 = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 16),
            nn.GELU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.spp(x)
        x1 = self.mlp_1(x)
        x = self.mlp_2(x1)
        x = x[:, :, None]
        x = self.mlp_3(x)
        return x, x1

if __name__ == '__main__':
    x = torch.rand(1, 1, 320, 240)
    model = FeatureExtractor(1)
    print(model(x)[1].shape)
