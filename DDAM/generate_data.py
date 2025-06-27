import torch
import numpy as np
import os
from core.diffusion import GaussianDiffusion
# from core.transformer import ATransformer
from core.encoder import ATransformer
from core.datasets import data_loader
from torch import nn
import torch
from  torchvision import utils
from torchvision import transforms

class WriteData:
    def __init__(self, args):
        model = nn.DataParallel(ATransformer(in_dim=args.in_ch,
                                             condition_dim=args.context_dim,
                                             model_dim=args.time_embed,
                                             h_dim=args.hidden_dim,
                                             head=args.nums_head,
                                             mask=args.mask,
                                             layer_nums=args.layer_nums,
                                             re_embedding_dim=128), device_ids=args.device_ids)

        model.load_state_dict(torch.load(args.checkpoint_path))
        model.eval()
        model.cuda()

        self.diffusion = GaussianDiffusion(model=model, beta=args.beta, T=args.time, beta_schedule=args.beta_schedule,
                                           only_x_0=args.only_x_0, inference_step=args.infer_step,
                                           save_step=args.save_step,
                                           sample_mode=args.mode
                                           )

        self.write_path = args.write_path

        if not os.path.exists(self.write_path):
            os.makedirs(self.write_path)


        self.data_root = args.data_root

        self.nums_sample = args.batch_size
        self.input_shape = args.input_shape

    def write_data(self, data):
        #B,
        pass

    def generate_data(self):
        count = 0

        to_plt = transforms.ToPILImage()

        for image, x, re, alpha, cl, cd in data_loader(self.data_root, self.nums_sample, test=False, generate=True)[1]:

            image, x, re, alpha, cl, cd = image, x.float().cuda(), re.float().cuda(), \
                                      alpha.float().cuda(), cl.float().cuda(), cd.float().cuda()

            condition = [x, re, alpha]
            z = torch.cat([cl, cd], dim=-1)


            z_t = torch.randn_like(z)

            all_data = self.diffusion.p_sample(z_t, condition)

            r = re[:,None].repeat(1, x.shape[1], 1)
            a = alpha[:, None].repeat(1, x.shape[1], 1)
            all_data =all_data[:, None].repeat(1, x.shape[1], 1)

            d = torch.cat([x, r, a, all_data], -1)

            for i in range(d.shape[0]):
                image_0 = image[i]

                nd = d[i].cpu().numpy()
                if not os.path.exists(self.write_path+f'/NAVA{count}'):
                    os.makedirs(self.write_path+f'/NAVA{count}')
                np.savetxt(os.path.join(self.write_path+f'/NAVA{count}', f'{0}.txt'), nd)
                image_0 = to_plt(image_0)
                image_0.save(self.write_path+f'/NAVA{count}'+f'/{0}.png')


                count += 1

        return


if __name__ == '__main__':
    import argparse

    paser = argparse.ArgumentParser()

    paser.add_argument('--input_shape', default=2)
    paser.add_argument('--in_ch', default=2)
    paser.add_argument('--context_dim', default=128)
    paser.add_argument('--hidden_dim', default=256)
    paser.add_argument('--time_embed', default=128)
    paser.add_argument('--nums_head', default=4)
    paser.add_argument('--bias', default=False)
    paser.add_argument('--layer_nums', default=4)
    paser.add_argument('--drop_out', default=0.)

    paser.add_argument('--time', default=1000)
    paser.add_argument('--beta_schedule', default='linear')
    paser.add_argument('--beta', default=[0.0001, 0.02])
    paser.add_argument('--only_x_0', default=True)
    paser.add_argument('--infer_step', default=50)
    paser.add_argument('--save_step', default=1)

    paser.add_argument('--device', default='cuda')
    paser.add_argument('--device_ids', default=[0, 1])
    paser.add_argument('--batch_size', default=128)

    paser.add_argument('--data_root', default=r'AirfoilNet/data/NACA 4 digit airfoils')

    paser.add_argument('--checkpoint_path', default=r'model/model1000.pth')
    paser.add_argument('--write_path', default=r'NACA_DATA_ddim_step50')
    paser.add_argument('--mask', default=0.)
    paser.add_argument('--mode', default='ddim')
    args = paser.parse_args()

    w = WriteData(args)

    w.generate_data()