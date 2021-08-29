import numpy as np
import torch
import torch.nn as nn


class GaussianKernel(nn.Module):
    def __init__(self, size, type):
        super(GaussianKernel, self).__init__()

        s = (size - 1) // 2
        _x = torch.linspace(-s, s, size).reshape((size, 1)).repeat((1, size))
        _y = torch.linspace(-s, s, size).reshape((1, size)).repeat((size, 1))
        self.d = _x ** 2 + _y ** 2
        self.register_buffer('GaussianKernel_d', self.d)
        self.type = type
        assert self.type in ['normal', 'reciprocal']


    def forward(self, sigma):
        if self.type == 'normal':
            k = sigma ** 2 + 0.01
            A = 1. / (2. * np.pi * k)
            d = -1. / (2. * k) * self.d#.cuda()
            B = torch.exp(d)
            B = A * B
            return B
        elif self.type == 'reciprocal':
            k = sigma ** 2
            A = k / (2. * np.pi)
            
            d = -k / 2. * self.d#.cuda()
            B = torch.exp(d)
            B = A * B
            return B

class MassiveGaussianKernel(nn.Module):
    def __init__(self, size, type):
        super(MassiveGaussianKernel, self).__init__()

        s = (size - 1) // 2
        _x = torch.linspace(-s, s, size).reshape((size, 1)).repeat((1, size))#.cuda()
        _y = torch.linspace(-s, s, size).reshape((1, size)).repeat((size, 1))#.cuda()
        self.d = _x ** 2 + _y ** 2

        self.size = size
        self.type = type
        assert self.type in ['normal', 'reciprocal']

    def forward(self, sigma):
        if self.type == 'normal':
            raise NotImplementedError
        elif self.type == 'reciprocal':
            n, c, h, w = sigma.shape
            assert n == 1 and c == 1
            sigma = sigma.reshape((h, w, 1, 1))
            k = sigma ** 2
            A = k / (2. * np.pi)
            d = -k / 2. * self.d.reshape((1, 1, self.size, self.size))
            B = torch.exp(d)
            B = A * B
            return B

def _GaussianKernel(sigma, size):
    s = (size - 1) // 2
    A = 1. / (2. * np.pi * ((sigma + 0.1) ** 2))
    x = torch.linspace(-s, s, size).reshape((size, 1)).repeat((1, size))#.cuda()
    y = torch.linspace(-s, s, size).reshape((1, size)).repeat((size, 1))#.cuda()
    d = x ** 2 + y ** 2
    d = -1. / (2. * ((sigma + 0.1) ** 2)) * d
    B = torch.exp(d)
    B = A * B
    return B

if __name__ == '__main__':
    # sigma = 1.
    size = 11

    G = GaussianKernel(7, 'reciprocal')
    g = G(3.0)
    print(g)

    G = MassiveGaussianKernel(7, 'reciprocal')
    sigma_map = torch.ones((1, 1, 3, 3))#.cuda() * 3.0
    g = G(sigma_map)
    print(g)

    # def _inner_check(sigma):
    #     G = GaussianKernel(size, 'reciprocal')
    #     inp = torch.tensor(sigma)
    #     Gaussian = G(inp)
    #     Gaussian[5, 5] = 0
    #     Gaussian_ = _GaussianKernel(sigma, size)
    #
    #     from matplotlib import pyplot as plt
    #     plt.imshow(Gaussian.cpu())
    #     plt.colorbar()
    #     plt.show()
    #
    #     # print(sigma, Gaussian.sum())
    #     # print(Gaussian)
    #     # print(Gaussian_ - Gaussian)
    #
    # for sigma in [0.5, 1.0, 1.5]:
    #     # _inner_check(float(sigma))
    #     _inner_check(torch.tensor(float(sigma)))
