import torch.nn as nn
# from torch.hub import load_state_dict_from_url
import torch
import torch.nn.functional as F
from .gumbel_softmax import GumbleSoftmax
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, last_relu=True,patch_groups=1, 
                 base_scale=2, is_first=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes // self.expansion, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes // self.expansion)
        self.conv2 = nn.Conv2d(planes // self.expansion, planes // self.expansion, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes // self.expansion)
        self.conv3 = nn.Conv2d(planes // self.expansion, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.have_pool = False
        self.have_1x1conv2d = False

        self.first_downsample = nn.AvgPool2d(3, stride=2, padding=1) if (base_scale == 4 and is_first) else None
            
        if self.downsample is not None:
            self.have_pool = True
            if len(self.downsample) > 1:
                self.have_1x1conv2d = True
        
        self.stride = stride
        self.last_relu = last_relu

    def forward(self, x):
        if self.first_downsample is not None:
            x = self.first_downsample(x)
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.last_relu:
            out = self.relu(out)
        return out

    def forward_calc_flops(self, x):
        flops = 0
        if self.first_downsample is not None:
            x = self.first_downsample(x)
            _, c, h, w = x.shape
            flops += 9 * c * h * w
        #     print('This is the first bottleneck of a base branch')
        # print('In a base bottleneck, x shape: ', x.shape)
        residual = x
        c_in = x.shape[1]
        out = self.conv1(x)
        _,c_out,h,w = out.shape
        flops += c_in * c_out * h * w  / self.conv1.groups

        out = self.bn1(out)
        out = self.relu(out)

        c_in = c_out
        out = self.conv2(out)
        _,c_out,h,w = out.shape
        flops += c_in * c_out * h * w * 9 / self.conv2.groups

        out = self.bn2(out)
        out = self.relu(out)

        c_in = c_out
        out = self.conv3(out)
        _,c_out,h,w = out.shape
        flops += c_in * c_out * h * w / self.conv3.groups
        out = self.bn3(out)

        if self.downsample is not None:
            c_in = x.shape[1]
            residual = self.downsample(x)
            _, c, h, w = residual.shape
            if self.have_pool:
                flops += 9 * c_in * h * w
            if self.have_1x1conv2d:
                flops += c_in * c * h * w
        
        out += residual
        if self.last_relu:
            out = self.relu(out)
        return out, flops

class Bottleneck_refine(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, last_relu=True,patch_groups=1, base_scale=2, is_first = True):
        super(Bottleneck_refine, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes // self.expansion, kernel_size=1, bias=False,groups=patch_groups)
        self.bn1 = nn.BatchNorm2d(planes // self.expansion)
        self.conv2 = nn.Conv2d(planes // self.expansion, planes // self.expansion, kernel_size=3, stride=stride,
                               padding=1, bias=False,groups=patch_groups)
        self.bn2 = nn.BatchNorm2d(planes // self.expansion)
        self.conv3 = nn.Conv2d(planes // self.expansion, planes, kernel_size=1, bias=False,groups=patch_groups)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        self.stride = stride
        self.last_relu = last_relu
        self.patch_groups = patch_groups

    def forward(self, x, mask, inference=False):
        residual = x
        if self.downsample is not None:     # skip connection before mask
            residual = self.downsample(x)

            
        b,c,h,w = x.shape
        g = mask.shape[1]
        m_h = mask.shape[2]
        mask1 = mask.clone()
        if g > 1:
            mask1 = mask1.unsqueeze(1).repeat(1,c//g,1,1,1).transpose(1,2).reshape(b,c,m_h,m_h)
            
        mask1 = F.interpolate(mask1, size = (h,w))
        # print(mask1.shape, x.shape)
        out = x * mask1
        # print(mask1)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        c_out = out.shape[1]
        # print(mask1.shape, mask.shape)
        mask2 = mask
        if g > 1:
            mask2 = mask2.unsqueeze(1).repeat(1,c_out//g,1,1,1).transpose(1,2).reshape(b,c_out,m_h,m_h)

        mask2 = F.interpolate(mask2, size = (h,w))
        # print(mask2.shape, out.shape)
        out = out * mask2
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # print(mask2.shape, out.shape)
        out = out * mask2
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        if self.last_relu:
            out = self.relu(out)
        return out


    def forward_calc_flops(self, x, mask, inference=False):
        # print('refine bottleneck, input shape: ', x.shape)
        residual = x
        flops = 0
        # print('In a refine bottleneck, x shape: ', x.shape)
        if self.downsample is not None:     # skip connection before mask
            c_in = x.shape[1]
            residual = self.downsample(x)
            flops += c_in * residual.shape[1] * residual.shape[2] * residual.shape[3]

        b,c,h,w = x.shape
        g = mask.shape[1]
        m_h = mask.shape[2]
        ratio = mask.sum() / mask.numel()
        # ratio = 0.75
        mask1 = mask.clone()
        if g > 1:
            mask1 = mask1.unsqueeze(1).repeat(1,c//g,1,1,1).transpose(1,2).reshape(b,c,m_h,m_h)

        mask1 = F.interpolate(mask1, size = (h,w))
        # print(mask1.shape, x.shape)
        out = x * mask1
        c_in = out.shape[1]
        out = self.conv1(out)
        flops += ratio * c_in * out.shape[1] * out.shape[2] * out.shape[3] / self.conv1.groups
        out = self.bn1(out)
        out = self.relu(out)

        c_out = out.shape[1]
        # print(mask1.shape, mask.shape)
        mask2 = mask
        if g > 1:
            mask2 = mask2.unsqueeze(1).repeat(1,c_out//g,1,1,1).transpose(1,2).reshape(b,c_out,m_h,m_h)
        mask2 = F.interpolate(mask2, size = (h,w))
        out = out * mask2
        c_in = out.shape[1]
        out = self.conv2(out)
        flops += ratio * c_in * out.shape[1] * out.shape[2] * out.shape[3] * 9 / self.conv2.groups
        out = self.bn2(out)
        out = self.relu(out)

        out = out * mask2
        c_in = out.shape[1]
        out = self.conv3(out)
        flops += ratio * c_in * out.shape[1] * out.shape[2] * out.shape[3]  / self.conv3.groups
        out = self.bn3(out)
        out += residual
        if self.last_relu:
            out = self.relu(out)
        return out, flops
        
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
        self.fc_gs = nn.Conv2d(groups*4,groups*2,kernel_size=1,stride=1,padding=0,bias=True, groups = groups)
        self.fc_gs.bias.data[:2*groups:2] = 1.0
        self.fc_gs.bias.data[1:2*groups+1:2] = 10.0
        self.gs = GumbleSoftmax()

    def forward(self, x, temperature=1.0):
        gates = self.conv3x3_gs(x)
        gates = self.pool(gates)
        gates = self.fc_gs(gates)

        # print(gates)
        # assert(0==1)
        gates = gates.view(x.shape[0],self.groups,2,self.mask_size,self.mask_size)

        # for i in range(gates.shape[1]):
        #     print(gates[0,i,:,:,:])
        #     print('hhh')
        # 
        # print(temperature)
        gates = self.gs(gates, temp=temperature, force_hard=True)
        gates = gates[:,:,1,:,:]
        # for i in range(gates.shape[1]):
        #     print(gates[0,i,:,:])
        #     print('hhh')
        # assert(0==1)
        return gates

    def forward_calc_flops(self, x, temperature=1.0):
        flops = 0
        c_in = x.shape[1]
        gates = self.conv3x3_gs(x)
        flops += c_in * gates.shape[1] * gates.shape[2] * gates.shape[3] * 9 / self.groups

        flops += gates.shape[1] * gates.shape[2] * gates.shape[3]
        gates = self.pool(gates)

        c_in = gates.shape[1]
        # print('1.shape:', gates.shape, gates)
        gates = self.fc_gs(gates)
        # print('2.shape:', gates.shape, gates)
        flops += c_in * gates.shape[1] * gates.shape[2] * gates.shape[3] / self.groups
        gates = gates.view(x.shape[0],self.groups,2,self.mask_size,self.mask_size)
        # print(temperature)
        gates = self.gs(gates, temp=temperature, force_hard=True)
        # print('3.shape:', gates.shape, gates)
        gates = gates[:,:,1,:,:]
        # print('4.shape:', torch.mean(gates), gates.shape, gates)
        return gates, flops

class sarModule(nn.Module):
    def __init__(self, block_base, block_refine, in_channels, out_channels, blocks, stride, groups=1,mask_size=7, alpha=1,beta=1, base_scale=2):
        super(sarModule, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.patch_groups = groups
        self.mask_size = mask_size
        self.base_scale = base_scale
        assert(base_scale in [2,4])
        self.base_scale = base_scale
        mask_gen_list = []
        for _ in range(blocks - 1):
            mask_gen_list.append(maskGen(groups=groups,inplanes=out_channels//alpha,mask_size=mask_size))
        self.mask_gen = nn.ModuleList(mask_gen_list)
        base_last_relu = True if alpha > 1 else False
        refine_last_relu = True if beta > 1 else False
        self.base_module = self._make_layer(block_base, in_channels, out_channels// alpha, blocks - 1, 2, last_relu=base_last_relu, base_scale=base_scale)
        
        self.refine_module = self._make_layer(block_refine, in_channels, out_channels*beta , blocks - 1, 1, last_relu=refine_last_relu, base_scale=base_scale)
        self.alpha = alpha
        self.beta = beta
        if alpha > 1:
            self.base_transform = nn.Sequential(
                nn.Conv2d(out_channels // alpha, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        if beta > 1:
            self.refine_transform = nn.Sequential(
                nn.Conv2d(out_channels * beta, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.fusion = self._make_layer(block_base, out_channels, out_channels, 1, stride=stride, base_scale=base_scale)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, last_relu=True, base_scale=2):
        downsample = []
        if stride != 1:
            downsample.append(nn.AvgPool2d(3, stride=2, padding=1))
            # if self.base_scale == 2:
            #     downsample.append(nn.AvgPool2d(3, stride=2, padding=1))
            # else:
            #     downsample.append(nn.AvgPool2d(3, stride=2, padding=1))
            #     downsample.append(nn.AvgPool2d(3, stride=2, padding=1))
        if inplanes != planes:
            downsample.append(nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False))
            downsample.append(nn.BatchNorm2d(planes))
        # print(downsample)
        downsample = None if downsample == [] else nn.Sequential(*downsample)
        layers = []
        if blocks == 1:         # fuse, is not the first of a base branch
            layers.append(block(inplanes, planes, stride=stride, downsample=downsample,
                                patch_groups=self.patch_groups, base_scale=base_scale, is_first = False))
        else:
            layers.append(block(inplanes, planes, stride, downsample,patch_groups=self.patch_groups, 
                             base_scale=base_scale, is_first = True))
            for i in range(1, blocks):
                layers.append(block(planes, planes,
                                    last_relu=last_relu if i == blocks - 1 else True, 
                                    patch_groups=self.patch_groups, base_scale=base_scale, is_first = False))

        return nn.ModuleList(layers)

    def forward(self, x, temperature=1e-8, inference=False):
        _masks = []
        x_refine = x
        for i in range(len(self.base_module)):
            x_base = self.base_module[i](x_base) if i!=0 else self.base_module[i](x)
            mask = self.mask_gen[i](x_base, temperature=temperature)
            _masks.append(mask)
            x_refine = self.refine_module[i](x_refine, mask, inference=False) 
        if self.alpha > 1:
            x_base = self.base_transform(x_base)
        if self.beta > 1:
            x_refine= self.refine_transform(x_refine)
        _,_,h,w = x_refine.shape
        x_base = F.interpolate(x_base, size = (h,w))
        out = self.relu(x_base + x_refine)
        out = self.fusion[0](out)
        return out, _masks

    def forward_calc_flops(self, x, temperature=1e-8, inference=False):
        b,c,h,w = x.size()
        flops = 0
        _masks = []
        x_refine = x
        for i in range(len(self.base_module)):
            x_base, _flops = self.base_module[i].forward_calc_flops(x_base) if i!=0 else self.base_module[i].forward_calc_flops(x)
            flops += _flops
            mask, _flops = self.mask_gen[i].forward_calc_flops(x_base, temperature=temperature)
            _masks.append(mask)
            flops += _flops
            x_refine, _flops = self.refine_module[i].forward_calc_flops(x_refine, mask, inference=False)
            flops += _flops

        c = x_base.shape[1]
        _,_, h,w = x_refine.shape
        if self.alpha > 1:
            x_base = self.base_transform(x_base)
            flops += c * x_base.shape[1] * x_base.shape[2] * x_base.shape[3]
        if self.beta > 1:
            c_in = x_refine.shape[1]
            x_refine = self.refine_transform(x_refine)
            flops += c_in * x_refine.shape[1] * x_refine.shape[2] * x_refine.shape[3]
        x_base = F.interpolate(x_base, size = (h,w))
        out = self.relu(x_base + x_refine)
        out, _flops = self.fusion[0].forward_calc_flops(out)
        flops += _flops
        return out, _masks, flops

class sarResNet(nn.Module):
    def __init__(self, block_base, block_refine, layers, num_classes=1000, patch_groups=1, mask_size=7, width=1.0, alpha=1, beta=1, base_scale=2):
        num_channels = [int(64*width), int(128*width), int(256*width), 512]
        # print(num_channels)
        self.inplanes = 64
        super(sarResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, num_channels[0], kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels[0])
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = sarModule(block_base, block_refine, num_channels[0], num_channels[0]*block_base.expansion, 
                               layers[0], stride=2, groups=patch_groups,mask_size=mask_size, alpha=alpha, base_scale=base_scale)
        self.layer2 = sarModule(block_base, block_refine, num_channels[0]*block_base.expansion,
                               num_channels[1]*block_base.expansion, layers[1], stride=2, groups=patch_groups,mask_size=mask_size, alpha=alpha,beta=beta, base_scale=base_scale)
        
        self.layer3 = sarModule(block_base, block_refine, num_channels[1]*block_base.expansion,
                               num_channels[2]*block_base.expansion, layers[2], stride=2, groups=patch_groups,mask_size=2, alpha=alpha, beta=beta, base_scale=2)
        self.layer4 = self._make_layer(
            block_base, num_channels[2]*block_base.expansion, num_channels[3]*block_base.expansion, layers[3], stride=1)
        self.gappool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_channels[3]*block_base.expansion, num_classes)

        for k, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if 'gs' in str(k):
                    m.weight.data.normal_(0, 0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each block.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = []
        if stride != 1:
            downsample.append(nn.AvgPool2d(3, stride=2, padding=1))
        if inplanes != planes:
            downsample.append(nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False))
            downsample.append(nn.BatchNorm2d(planes))
        downsample = None if downsample == [] else nn.Sequential(*downsample)

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        for _ in range(1, blocks):
            layers.append(block(planes, planes))

        return nn.ModuleList(layers)

    def forward(self, x, temperature=1.0, inference=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # print('before layer 1:', x.shape)
        _masks = []
        x, mask = self.layer1(x, temperature=temperature, inference=inference)
        _masks.extend(mask)

        x, mask = self.layer2(x, temperature=temperature, inference=inference)
        _masks.extend(mask)
        # print('before layer 3:', x.shape)
        x, mask = self.layer3(x, temperature=temperature, inference=inference)
        _masks.extend(mask)
        # print('before layer 4:', x.shape)
        for i in range(len(self.layer4)):
            x = self.layer4[i](x)

        x = self.gappool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, _masks

    def forward_calc_flops(self, x, temperature=1.0, inference=False):
        flops = 0
        c_in = x.shape[1]
        x = self.conv1(x)
        flops += c_in * x.shape[1] * x.shape[2] * x.shape[3] * self.conv1.weight.shape[2]*self.conv1.weight.shape[3]
        x = self.bn1(x)
        x = self.relu(x)

        _masks = []
        x, mask, _flops = self.layer1.forward_calc_flops(x, temperature=temperature, inference=inference)
        _masks.extend(mask)
        flops += _flops
        
        x, mask, _flops = self.layer2.forward_calc_flops(x, temperature=temperature, inference=inference)
        _masks.extend(mask)
        flops += _flops

        x, mask, _flops = self.layer3.forward_calc_flops(x, temperature=temperature, inference=inference)
        flops += _flops
        _masks.extend(mask)

        for i in range(len(self.layer4)):
            x, _flops = self.layer4[i].forward_calc_flops(x)
            flops += _flops

        flops += x.shape[1] * x.shape[2] * x.shape[3]
        x = self.gappool(x)
        x = x.view(x.size(0), -1)
        c_in = x.shape[1]
        x = self.fc(x)
        flops += c_in * x.shape[1]

        return x, _masks, flops

def sar_resnet_cifar_alphaBase(depth, num_classes=10, patch_groups=1, mask_size=4, width=1.0, alpha=1, beta=1, base_scale=2):
    layers = {
        50: [3, 4, 6, 3],
        101: [4, 8, 18, 3],
        152: [5, 12, 30, 3]
    }[depth]
    model = sarResNet(block_base=Bottleneck, block_refine=Bottleneck_refine, layers=layers, 
                      num_classes=num_classes, patch_groups=patch_groups, mask_size=mask_size, width=width, 
                      alpha=alpha, beta=beta, base_scale=base_scale)
    return model

def sar_resnet50_alphaBase_4stage_cifar(args):
    return sar_resnet_cifar_alphaBase(depth=50, num_classes=args.num_classes, patch_groups=args.patch_groups, mask_size=args.mask_size, alpha=args.alpha, beta=args.beta, base_scale=args.base_scale)


if __name__ == "__main__":
    import argparse
    from op_counter import measure_model
    parser = argparse.ArgumentParser(description='PyTorch SARNet')
    args = parser.parse_args()
    args.num_classes = 100
    args.patch_groups = 2
    args.mask_size = 4
    args.alpha = 2
    args.beta = 1
    args.base_scale = 2
    sar_res = sar_resnet50_alphaBase_4stage_cifar(args)
    with torch.no_grad():
        
        print(sar_res)
        x = torch.rand(1,3,32,32)
        sar_res.eval()

        y1, _masks, flops = sar_res.forward_calc_flops(x,inference=False,temperature=1e-8)
        # print(len(_masks))
        # for i in range(len(_masks)):
        #     print(_masks[i])
        print(flops / 1e9)
    