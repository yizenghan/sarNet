import torch
import torch.nn as nn
import torch.nn.functional as F


class Gumbel(nn.Module):
    ''' 
    Returns differentiable discrete outputs. Applies a Gumbel-Softmax trick on every element of x. 
    '''
    def __init__(self, eps=1e-8):
        super(Gumbel, self).__init__()
        self.eps = eps

    def forward(self, x, temperature=1.0, gumbel_noise=True):
        if not self.training:  # no Gumbel noise during inference
            return (x >= 0).float()

        if gumbel_noise:
            eps = self.eps
            U1, U2 = torch.rand_like(x), torch.rand_like(x)
            g1, g2 = -torch.log(-torch.log(U1 + eps)+eps), - \
                torch.log(-torch.log(U2 + eps)+eps)
            x = x + g1 - g2

        soft = torch.sigmoid(x / temperature)
        hard = ((soft >= 0.5).float() - soft).detach() + soft
        assert not torch.any(torch.isnan(hard))
        return hard

       
class maskGen(nn.Module):
    def __init__(self, groups=1, inplanes=64, mask_size=7):
        super(maskGen,self).__init__()
        self.groups = groups
        self.mask_size = mask_size
        self.conv3x3_gs = nn.Sequential(
            nn.Conv2d(inplanes, groups*4,kernel_size=3, padding=1, stride=1, bias=False, groups = groups),
            nn.BatchNorm2d(groups*4),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool2d((mask_size,mask_size))
        self.fc_gs = nn.Conv2d(groups*4,groups,kernel_size=1,stride=1,padding=0,bias=True, groups = groups)
        self.fc_gs.bias.data[:] = 5.0
        self.gs = Gumbel()

    def forward(self, x, temperature=1.0):
        gates = self.conv3x3_gs(x)
        gates = self.pool(gates)
        gates = self.fc_gs(gates)
        # print(gates.shape)
        gates = gates.view(x.shape[0],self.groups,self.mask_size,self.mask_size)
        # print(gates.shape)
        # print(temperature)
        gates = self.gs(gates, temperature=temperature)
        return gates

    def forward_calc_flops(self, x, temperature=1.0):
        flops = 0
        c_in = x.shape[1]
        gates = self.conv3x3_gs(x)
        flops += c_in * gates.shape[1] * gates.shape[2] * gates.shape[3] * 9 / self.groups

        flops += gates.shape[1] * gates.shape[2] * gates.shape[3]
        gates = self.pool(gates)

        c_in = gates.shape[1]
        gates = self.fc_gs(gates)
        flops += c_in * gates.shape[1] * gates.shape[2] * gates.shape[3] / self.groups
        gates = gates.view(x.shape[0],self.groups,self.mask_size,self.mask_size)
        # print(temperature)
        gates = self.gs(gates, temperature=temperature)


if __name__ == '__main__':
    x = torch.rand(1,64,9,9)
    mask_gen = maskGen(groups=4, inplanes=64, mask_size=3)

    mask = mask_gen(x, temperature=10.0)
    print(mask)