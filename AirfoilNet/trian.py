from core.net import MLPNet, AGRUNet
from core.test import TGRUNet
from torch.optim import Adam, AdamW, SGD
from torch.optim import lr_scheduler
from datasets import data_loader, evalu
from torch import nn
import os
import torch
from tqdm import tqdm
from torch.nn import functional as F
import random
import time

random.seed(1234)
torch.manual_seed(1234)

VAL_SUQ = 10

def seq_lossfc(out_list, target, gamma=0.85):
    outlen = len(out_list)
    loss = 0.

    for i in range(outlen):

        i_weight = gamma**(outlen - i - 1)
        i_loss = i_weight * F.l1_loss(out_list[i], target)
        loss += i_loss

    return loss


def train_model(args):
    # model = nn.DataParallel(Compose(inc=args.in_ch,
    #              outc=6, layers=args.layers, drop_out=args.drop_out), device_ids=args.device_ids)
    if args.predicted_head == 'gru':
        model = nn.DataParallel(TGRUNet(cnn_in_ch=args.in_ch, cnn_out_ch=args.out_ch,
                                    spp_layers=args.spp_layers, geometry_dim=args.x_dim, re_embedding_dim=args.re_embedding_dim, gru_layers=args.gru_layers, out_ch=2),
                                device_ids=args.device_ids)
    elif args.predicted_head == 'mlp':
        model = nn.DataParallel(MLPNet(cnn_in_ch=args.in_ch, cnn_out_ch=args.out_ch,
                                       spp_layers=args.spp_layers, geometry_dim=args.x_dim, re_embedding_dim=args.re_embedding_dim),
                                device_ids=args.device_ids)
    else:
        model = nn.DataParallel(AGRUNet(),
                                device_ids=args.device_ids)

    model.cuda()
    if args.checkpoint != '':
        check_point = torch.load(args.checkpoint)
    # check_point = {k.replace('module.', ''): v for k, v in check_point.items()}

        model.load_state_dict(check_point)
    epoch = args.epoch

    dataloader = data_loader(args.root, args.batch_size, test=False)

    testloader = data_loader(args.root, 1, test=True)
    start = time.time()
    evalu(testloader, model)
    end = time.time()
    print((end-start))

    optimizer = AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay, eps=args.eps)

    scheduler = lr_scheduler.OneCycleLR(optimizer, args.lr, len(dataloader)*epoch + 100,
                                              pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    model.train()
    best_val = 10000.
    print(f'Paramters:{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    save_path = ''
    for i in range(epoch):
        total_loss = 0.
        total_loss1 = 0.
        total_loss2 = 0.
        total_loss3 = 0.
        total_num = 0.
        with tqdm(dataloader, dynamic_ncols=True) as tqdmloader:
            for image, x, re, alpha, cl, cd in tqdmloader:
                optimizer.zero_grad()

                image, x, re, alpha, cl, cd = image.float().cuda(), x.float().cuda(), re.float().cuda(), \
                                              alpha.float().cuda(), cl.float().cuda(), cd.float().cuda()

                # condition = torch.cat([re, alpha], dim=-1)


                #"gru_" means initialize C0 with the output of MLP
                if args.predicted_head == 'gru_':
                    cnn_pre, out_put, mlp_out = model(image, x, re, alpha)
                    loss1 = F.l1_loss(cnn_pre, x)
                    loss2 = seq_lossfc(out_put, torch.cat([cl, cd], dim=-1))
                    loss3 = F.l1_loss(mlp_out, torch.cat([cl, cd], dim=-1))
                elif args.predicted_head == 'gru':
                    cnn_pre, out_put = model(image, x, re, alpha)
                    loss1 = F.l1_loss(cnn_pre, x)
                    loss2 = seq_lossfc(out_put, torch.cat([cl, cd], dim=-1))

                else:
                    cnn_pre, out_put = model(image, x, re, alpha)
                    loss1 = F.l1_loss(cnn_pre, x)
                    loss2 = F.l1_loss(out_put, torch.cat([cl, cd], dim=-1))

                loss = loss1 + loss2

                loss.backward()
                optimizer.step()

                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

                scheduler.step()

                total_loss += loss.item()
                total_num += 1

                total_loss1 += loss1.item()
                total_loss2 += loss2.item()
                # if "agru" you should turn on the code blow:
                # total_loss3 += loss3.item()

                tqdmloader.set_description(f"Epoch: {i}")
                tqdmloader.set_postfix(ordered_dict={
                    'train_loss': total_loss / total_num,
                    'loss_1': total_loss1 / total_num,
                    'loss_2': total_loss2 / total_num,
                    # if "agru" you should turn on the code blow:
                    # 'loss_3': total_loss3 / total_num,
                    'lr': optimizer.state_dict()['param_groups'][0]['lr']
                })
        if i % VAL_SUQ == 0:
            loss_ = evalu(testloader, model)
            # if best_val > loss_:
            save_path = args.save_path + f'/{i}_{args.predicted_head}_model{loss_}_{args.gru_layers}.pth'
            #     best_val = loss_
            #
            # if not os.path.exists(args.save_path):
            #     os.makedirs((args.save_path))

            torch.save(model.state_dict(), save_path)





if __name__ == '__main__':
    import argparse
    paser = argparse.ArgumentParser()

    paser.add_argument('--in_ch', default=1)
    paser.add_argument('--out_ch', default=2)
    paser.add_argument('--spp_layers', default=4)
    paser.add_argument('--x_dim', default=69)
    paser.add_argument('--device', default='cuda')
    paser.add_argument('--device_ids', default=[0, ])
    paser.add_argument('--batch_size', default=256)
    paser.add_argument('--lr', default=0.00025)
    # paser.add_argument('--lr', default=0.0004)
    paser.add_argument('--root', default=r'data/NACA 4 digit airfoils')
    paser.add_argument('--drop_out', default=0.1)
    paser.add_argument('--epoch', default=1000)
    paser.add_argument('--weight_decay', default=0.0001)
    paser.add_argument('--clip_grad', default=1.)
    paser.add_argument('--save_path', default=r'.')
    paser.add_argument('--predicted_head', default='mlp')
    paser.add_argument('--re_embedding_dim', default=32)
    # paser.add_argument('--re_embedding_dim', default=1)
    paser.add_argument('--gru_layers', default=12)
    paser.add_argument('--eps', default=1e-8)
    paser.add_argument('--checkpoint', default='')


    args = paser.parse_args()

    train_model(args)