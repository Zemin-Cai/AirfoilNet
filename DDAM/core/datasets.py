import os
import torch
from torch.utils.data import Dataset, DataLoader
import csv
from glob import glob
import numpy as np
import random
from torch.nn import functional as F
from PIL import Image
from torchvision import transforms
import math

random.seed(1234)


def padding(x, max_lenght):
    pad_size = max_lenght - x.shape[0]

    return F.pad(x, (0, 0, 0, pad_size))

def norm_re(x, min=50000, max=1000000, avr=420709.0736040609):
    return (x) / (max)


def re_embedding(timesteps, dim, max_period=1000000):
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
    args = timesteps.float() * freqs
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class ADatasets(Dataset):
    def __init__(self, root, symmetrica=False, test=False, generate=False):
        super().__init__()
        self.root = root
        self.symmetrica = symmetrica
        self.test = test

        self.data_list = glob(os.path.join(root, 'NACA*/*.txt'))

        self.pad = 69
        if self.symmetrica:
            self.symmetrica_list = os.path.join(root, 'Symmetrical airfoils')
            self.symmetrica_list = glob(os.path.join(self.symmetrica_list, 'NACA*/*.txt'))
            self.data_list += self.symmetrica_list
            self.pad = 131

        # train_nums = int(len(self.data_list) * 0.8)
        self.train_list = self.data_list
        self.test_list = glob(os.path.join(root, 'TNACA*/*.txt'))
        # self.train_list = random.sample(self.data_list, train_nums)
        # self.test_list = [e for e in self.data_list if e not in self.train_list]
        if test:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.RandomRotation((-45, 90)),
                transforms.ToTensor()
            ])

        self.generate = generate

        self.g_transform = transforms.Compose([
                transforms.ToTensor()
            ])

    def __getitem__(self, index):
        if not self.test:
            image_path = self.train_list[index]
            directory, file_name = os.path.split(image_path)
            image = Image.open(os.path.join(directory, '0.png'))
            if self.generate is False:

                image = self.transform(image)
            else:
                image = self.g_transform(image)

            data = np.loadtxt(self.train_list[index])
            data = torch.from_numpy(data)


            x_y = data[:, :2]
            r = data[0, 2]
            alpha = data[0, 3]
            cl = data[0, 4]
            cd = data[0, 5]

            x_y = padding(x_y, max_lenght=self.pad)
            # x_y = torch.mean(x_y, dim=0)


            # r = norm_re(r, avr=423309.0736040609)

            return image, x_y, r[None], alpha[None], cl[None], cd[None]

        else:
            image_path = self.test_list[index]
            directory, file_name = os.path.split(image_path)
            image = Image.open(os.path.join(directory, '0.png'))
            image = self.transform(image)


            data = np.loadtxt(self.test_list[index])
            data = torch.from_numpy(data)

            x_y = data[:, :2]
            r = data[0, 2]
            alpha = data[0, 3]
            cl = data[0, 4]
            cd = data[0, 5]


            x_y = padding(x_y, max_lenght=self.pad)

            # r = norm_re(r, avr=423309.0736040609)

            return image, x_y, r[None], alpha[None], cl[None], cd[None]

    def __len__(self):
        if self.test:
            return len(self.test_list)
        else:
            return len(self.train_list)

    def read_xy(self, file):

        a = np.loadtxt(file)

        return a

    def read_airfoil_csv(self, file):
        re = []
        alpha = []
        cl = []
        cd = []
        flag = False

        with open(file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if 'Reynolds number' in row:
                    re.append(float(row[1]))
                    continue
                if 'Alpha' in row:
                    flag = True
                    continue
                if flag:
                    alpha.append(float(row[0]))
                    cl.append(float(row[1]))
                    cd.append(float(row[2]))

        return re, alpha, cl, cd

    def save(self):
        for i in self.data_list:
            count = 0
            xy_path = glob(os.path.join(i, '*.dat'))
            xy = self.read_xy(xy_path[0])
            for j in glob(os.path.join(i, '*.csv')):
                re, alpha, cl, cd = self.read_airfoil_csv(j)
                re = re * len(alpha)
                for r, a, l, d in zip(re, alpha, cl, cd):

                    r = np.array(r).reshape(1, 1).repeat(xy.shape[0], axis=0)
                    a = np.array(a).reshape(1, 1).repeat(xy.shape[0], axis=0)
                    l = np.array(l).reshape(1, 1).repeat(xy.shape[0], axis=0)
                    d = np.array(d).reshape(1, 1).repeat(xy.shape[0], axis=0)

                    np.savetxt(os.path.join(i, f'{count}.txt'), np.concatenate([xy, r, a, l, d], axis=-1))
                    count += 1


def data_loader(root, batch_size, test, generate=False):
    data = ADatasets(root, symmetrica=False, test=test, generate=generate)
    data_five = ADatasets(r'AirfoilNet/data/NACA 5 digit airfoils', symmetrica=False, test=test, generate=generate)
    data = data + data_five
    print('Data Nums:', len(data))
    return len(data), DataLoader(data, batch_size=batch_size, pin_memory=4, shuffle=True)


def evalu(data, model, test=False):
    model.eval()
    total_loss1 = 0.
    total_loss2 = 0.
    total_num = 0
    total_num_ = 0.
    relative_error = 0.


    for image, x, re, alpha, cl, cd in data:
        image, x, re, alpha, cl, cd = image.float().cuda(), x.float().cuda(), re.float().cuda(), \
                                      alpha.float().cuda(), cl.float().cuda(), cd.float().cuda()

        cnn_pre, out_put = model(image=image, geometry=x, re=re, alpha=alpha, test_mode=True)

        loss1 = F.mse_loss(cnn_pre, x)
        loss2 = F.mse_loss(out_put, torch.cat([cl, cd], dim=-1))

        if test:
            print('orignal:', torch.cat([cl, cd], dim=-1).cpu().numpy(), 'prediction:', out_put.detach().cpu().numpy())

        total_loss1 += loss1.item()
        total_loss2 += loss2.item()
        total_num += 1


        a = F.l1_loss(out_put.detach(), torch.cat([cl, cd], dim=-1), reduction='none')
        b = out_put.detach().abs()
        c = a / b
        c = torch.mean(c, dim=0)
        relative_error += c



    print(f'eval_loss1:{total_loss1/total_num}   eval_loss2:{total_loss2/total_num}')
    print(f'relative_error:{relative_error/total_num}')
    return total_loss2/total_num

if __name__ == '__main__':
    testloader = data_loader('AirfoilNet/data/NACA 4 digit airfoils', 64, test=True)
    for i in testloader:
        print(i[1].shape)







