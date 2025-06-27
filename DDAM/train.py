import torch
import numpy as np
from core.diffusion import GaussianDiffusion
from core.datasets import data_loader
from torch.optim import Adam, AdamW
from torch.optim import lr_scheduler
from tqdm import tqdm
# from core.transformer import ATransformer
from core.encoder import ATransformer
from torch import nn
import os

def train_model(args):

    # model = nn.DataParallel(ATransformer(in_ch=args.in_ch,
    #              context_dim=args.context_dim,
    #              time_embed=args.time_embed,
    #              hidden_dim=args.hidden_dim,
    #              nums_head=args.nums_head,
    #              bias=False,
    #              layer_nums=args.layer_nums,
    #              re_embedding_dim=128), device_ids=args.device_ids)
    model = nn.DataParallel(ATransformer(in_dim=args.in_ch,
                                         condition_dim=args.context_dim,
                                         model_dim=args.time_embed,
                                         h_dim=args.hidden_dim,
                                         head=args.nums_head,
                                         mask=args.mask,
                                         layer_nums=args.layer_nums,
                                         re_embedding_dim=128), device_ids=args.device_ids)

    model.train()
    model.cuda()

    print(f'paramters:{sum([p.numel() for p in model.parameters() if p.requires_grad])}')

    if args.load_checkpoint:
        print('model loaded')
        model.load_state_dict(torch.load(args.load_checkpoint))

    diffusion = GaussianDiffusion(model, args.beta, args.time, args.beta_schedule)

    epoch = args.epoch
    batch_size = args.batch_size
    data_len, dataloader = data_loader(args.root, args.batch_size, test=False)
    step_size = data_len // batch_size if data_len % batch_size == 0 else data_len // batch_size + 1
    step_size = epoch * step_size

    optimizer = AdamW(diffusion.parameters(), args.lr, weight_decay=args.weight_decay)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=50*step_size, gamma=args.gamma)
    scheduler = lr_scheduler.OneCycleLR(optimizer, args.lr, step_size + 100,
                                              pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    diffusion.train()

    for i in range(epoch):
        total_loss = 0.
        total_num = 0.
        with tqdm(dataloader, dynamic_ncols=True) as tqdmloader:
            for image, x, re, alpha, cl, cd in tqdmloader:
                optimizer.zero_grad()

                _, x, re, alpha, cl, cd = image.float().cuda(), x.float().cuda(), re.float().cuda(), \
                                              alpha.float().cuda(), cl.float().cuda(), cd.float().cuda()


                x_0 = torch.cat([cl, cd], dim=-1)
                condition = [x, re, alpha]
                loss = diffusion.q_sample_loss(x_0=x_0, condition=condition)

                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                total_num += 1

                tqdmloader.set_description(f"Epoch: {i}")
                tqdmloader.set_postfix(ordered_dict={
                    'train_loss': total_loss / total_num,
                    'lr': optimizer.state_dict()['param_groups'][0]['lr']
                })

        if not os.path.exists(args.save_path):
            os.makedirs((args.save_path))
        if i % 100 == 0:
            save_path = args.save_path + f'{100+i}.pth'
            torch.save(model.state_dict(), save_path)



if __name__ == '__main__':
    import argparse
    paser = argparse.ArgumentParser()

    paser.add_argument('--in_ch', default=2)
    paser.add_argument('--context_dim', default=128)
    paser.add_argument('--hidden_dim', default=256)
    paser.add_argument('--time_embed', default=128)
    paser.add_argument('--nums_head', default=4)
    paser.add_argument('--bias', default=False)
    paser.add_argument('--layer_nums', default=4)
    paser.add_argument('--time', default=1000)
    paser.add_argument('--beta_schedule', default='linear')
    paser.add_argument('--beta', default=[0.0001, 0.02])
    paser.add_argument('--device', default='cuda')
    paser.add_argument('--device_ids', default=[0, 1])
    paser.add_argument('--batch_size', default=128)
    paser.add_argument('--lr', default=0.0002)
    paser.add_argument('--root', default=r'AirfoilNet/data/NACA 4 digit airfoils')
    paser.add_argument('--drop_out', default=0.)
    paser.add_argument('--epoch', default=1000)
    paser.add_argument('--weight_decay', default=0.0001)
    paser.add_argument('--gamma', default=0.85)
    paser.add_argument('--clip_grad', default=1.)
    paser.add_argument('--save_path', default=r'model')
    paser.add_argument('--load_checkpoint', default=None)
    paser.add_argument('--mask', default=0.)

    args = paser.parse_args()

    train_model(args)






