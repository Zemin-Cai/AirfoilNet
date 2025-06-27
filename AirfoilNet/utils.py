import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image

def save_airfoil(root):
    plt.figure()
    for p in os.listdir(root):
        deep = os.path.join(root, p)
        if glob(os.path.join(deep, '*.dat')):
            data = np.loadtxt(glob(os.path.join(deep, '*.dat'))[0])

            plt.subplot(1, 1, 1)
            plt.plot(data[:, 0], data[:, 1])
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(f'{deep}/0.png')
            image = Image.open(f'{deep}/0.png').convert('L')
            new_size = (image.width//2, image.height//2)
            image = image.resize(new_size, Image.ANTIALIAS)

            image.save(f'{deep}/0.png')

            plt.show()

    root = os.path.join(root, 'Symmetrical airfoils')
    for p in os.listdir(root):
        deep = os.path.join(root, p)
        if glob(os.path.join(deep, '*.dat')):
            data = np.loadtxt(glob(os.path.join(deep, '*.dat'))[0])


            plt.subplot(1, 1, 1)
            plt.plot(data[:, 0], data[:, 1])
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(f'{deep}/0.png')
            image = Image.open(f'{deep}/0.png').convert('L')
            new_size = (image.width//2, image.height//2)
            image = image.resize(new_size, Image.ANTIALIAS)
            image.save(f'{deep}/0.png')

