from datasets import evalu, data_loader
from core.net import GRUNet, MLPNet
from torch import nn
import argparse
import torch
from torch.nn import functional as F


paser = argparse.ArgumentParser()

paser.add_argument('--in_ch', default=1)
paser.add_argument('--out_ch', default=16)
paser.add_argument('--spp_layers', default=4)
paser.add_argument('--x_dim', default=69)
paser.add_argument('--device', default='cuda')
paser.add_argument('--device_ids', default=[0, 1])
paser.add_argument('--batch_size', default=1)

paser.add_argument('--root', default=r'data/NACA 4 digit airfoils')
paser.add_argument('--predicted_head', default='gru')


args = paser.parse_args()

if args.predicted_head == 'gru':
    model = nn.DataParallel(GRUNet(cnn_in_ch=args.in_ch, cnn_out_ch=args.out_ch,
                                spp_layers=args.spp_layers, geometry_dim=args.x_dim),
                            device_ids=args.device_ids)
else:
    model = nn.DataParallel(MLPNet(cnn_in_ch=args.in_ch, cnn_out_ch=args.out_ch,
                                   spp_layers=args.spp_layers, geometry_dim=args.x_dim),
                            device_ids=args.device_ids)

model.cuda()
model.load_state_dict(torch.load('AirfoilNet/models/*.pth'))

test_dataloader = data_loader(args.root, args.batch_size, test=True)

evalu(test_dataloader, model, False)


