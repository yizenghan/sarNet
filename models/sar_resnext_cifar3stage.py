import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
import torch
from .gumbel_softmax import GumbleSoftmax

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.reduction = reduction
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel // reduction), channel, bias=False),
            nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b,c)
        attention = self.fc(y)
        return x * attention.view(b,c,1,1)
    
    def forward_calc_flops(self, x):
        b, c, h, w = x.size()
        flops = c*h*w
        y = self.avg_pool(x).view(b,c)
        attention = self.fc(y)
        flops += c*c//self.reduction*2 + c
        return x * attention.view(b,c,1,1), flops

class ResNeXtBottleneck(nn.Module):
    expansion = 4
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """
    def __init__(self, inplanes, planes, cardinality, base_width, stride=1, downsample=None, 
                last_relu=True, patch_groups=1, base_scale=2, is_first=False):
        super(ResNeXtBottleneck, self).__init__()

        D = int(math.floor(planes * (base_width/64.0)))
        C = cardinality
        # print(cardinality)
        self.conv1 = nn.Conv2d(inplanes, D*C, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(D*C)

        self.conv2 = nn.Conv2d(D*C, D*C, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(D*C)

        self.conv3 = nn.Conv2d(D*C, planes*4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        self.have_pool = False
        self.have_1x1conv2d = False
        # self.is_first = is_first
        self.first_downsample = nn.AvgPool2d(3, stride=2, padding=1) if (base_scale == 4 and is_first) else None
        if self.downsample is not None:
            self.have_pool = True
            if len(self.downsample) > 1:
                self.have_1x1conv2d = True

        self.last_relu = last_relu

    def forward(self, x):
        residual = x
        
        
        if self.first_downsample is not None:
            x = self.first_downsample(x)

        out = self.conv1(x)
        out = self.relu(self.bn1(out))

        out = self.conv2(out)
        out = self.relu(self.bn2(out))

        out = self.conv3(out)
        out = self.bn3(out)
        

        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        if self.last_relu:
            out = self.relu(out)
        return out

    def forward_calc_flops(self, x):
        flops = 0
        residual = x
        # print(x.shape)
        if self.first_downsample is not None:
            x = self.first_downsample(x)
            _, c, h, w = x.shape
            flops += 9 * c * h * w
        
        
        c_in = x.shape[1]
        out = self.conv1(x)
        # print(out.shape)
        _,c_out,h,w = out.shape
        flops += c_in * c_out * h * w  / self.conv1.groups

        out = self.bn1(out)
        out = self.relu(out)

        c_in = c_out
        out = self.conv2(out)
        # print(out.shape)
        _,c_out,h,w = out.shape
        flops += c_in * c_out * h * w * 9 / self.conv2.groups

        out = self.bn2(out)
        out = self.relu(out)

        c_in = c_out
        out = self.conv3(out)
        # print(out.shape)
        _,c_out,h,w = out.shape
        flops += c_in * c_out * h * w / self.conv3.groups
        out = self.bn3(out)

        if self.downsample is not None:
            # print(self.downsample)
            # assert(0==1)
            c_in = x.shape[1]
            residual = self.downsample(x)
            _, c, h, w = residual.shape
            if self.have_pool:
                flops += 9 * c_in * h * w
            if self.have_1x1conv2d:
                flops += c_in * c * h * w
        # print(residual.shape)
        out += residual
        if self.last_relu:
            out = self.relu(out)
        return out, flops

class ResNeXtBottleneck_refine(nn.Module):
    expansion = 4
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """
    def __init__(self, inplanes, planes, cardinality, base_width, stride=1, downsample=None, 
                last_relu=True, patch_groups=1, base_scale=2, is_first=False):
        super(ResNeXtBottleneck_refine, self).__init__()

        D = int(math.floor(planes * (base_width/64.0)))
        C = cardinality
        # print(cardinality)
        self.conv1 = nn.Conv2d(inplanes, D*C, kernel_size=1, stride=1, padding=0, bias=False, groups=patch_groups)
        self.bn1 = nn.BatchNorm2d(D*C)

        self.conv2 = nn.Conv2d(D*C, D*C, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(D*C)

        self.conv3 = nn.Conv2d(D*C, planes*4, kernel_size=1, stride=1, padding=0, bias=False, groups=patch_groups)
        self.bn3 = nn.BatchNorm2d(planes*4)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        

        self.last_relu = last_relu

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
        mask2 = mask.clone()
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
        mask2 = mask.clone()
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
    def __init__(self, groups=1, inplanes=16, mask_size=4):
        super(maskGen,self).__init__()
        self.groups = groups
        self.mask_size = mask_size
        self.conv3x3_gs = nn.Sequential(
            nn.Conv2d(int(inplanes), int(groups*4), kernel_size=3, padding=1, stride=1, bias=False, groups = groups),
            nn.BatchNorm2d(groups*4),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool2d((mask_size,mask_size))
        self.fc_gs = nn.Conv2d(groups*4,groups*2,kernel_size=1,stride=1,padding=0,bias=True, groups = groups)
        self.fc_gs.bias.data[:2 * groups:2] = 1.0
        self.fc_gs.bias.data[1:2 * groups + 1:2] = 10.0
        self.gs = GumbleSoftmax()

    def forward(self, x, temperature=1.0):
        gates = self.conv3x3_gs(x)
        gates = self.pool(gates)
        gates = self.fc_gs(gates)
        gates = gates.view(x.shape[0], self.groups, 2, self.mask_size, self.mask_size)
        # print(temperature)
        gates = self.gs(gates, temp=temperature, force_hard=True)
        gates = gates[:, :, 1, :, :]
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
        gates = gates.view(x.shape[0], self.groups, 2, self.mask_size, self.mask_size)
        gates = self.gs(gates, temp=temperature, force_hard=True)
        gates = gates[:, :, 1, :, :]
        return gates, flops

class sarModule(nn.Module):
    def __init__(self, block_base, block_refine, in_channels, out_channels, 
                 blocks, stride, cardinality, base_width,
                 groups=1,
                 mask_size=4, alpha=1, beta=2, base_scale=2, fuse_se=False):
        super(sarModule, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.patch_groups = groups
        self.mask_size = mask_size
        self.base_scale = base_scale
        self.cardinality = cardinality
        self.base_width = base_width
        assert (base_scale in [2, 4])
        self.base_scale = base_scale
        mask_gen_list = []
        for _ in range(blocks - 1):
            mask_gen_list.append(maskGen(groups=groups,inplanes=out_channels*4 // alpha,mask_size=mask_size))
        self.mask_gen = nn.ModuleList(mask_gen_list)
        base_last_relu = True if alpha > 1 else False
        refine_last_relu = True if beta > 1 else False
        self.base_module = self._make_layer(block_base, in_channels, out_channels // alpha, blocks - 1, 2, last_relu=base_last_relu, base_scale=base_scale)
        self.refine_module = self._make_layer(block_refine, in_channels, out_channels * beta, blocks - 1, 1, last_relu=refine_last_relu, base_scale=base_scale)
        self.alpha = alpha
        self.beta = beta
        if alpha > 1:
            self.base_transform = nn.Sequential(
                nn.Conv2d(out_channels *4// alpha, out_channels*4, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels*4)
            )
        if beta > 1:
            self.refine_transform = nn.Sequential(
                nn.Conv2d(out_channels*4 * beta, out_channels*4, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels*4)
            )
        # print(fuse_se)
        self.fuse_se = fuse_se
        self.se_base = SELayer(channel=out_channels*4, reduction=16) if fuse_se else None
        self.se_refine = SELayer(channel=out_channels*4, reduction=16) if fuse_se else None
        self.fusion = self._make_layer(block_base, out_channels*4, out_channels, 1, stride=stride, base_scale=base_scale, is_fuse=True)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, last_relu=True, base_scale=2, is_fuse=False):
        downsample = []
        if stride != 1:
            downsample.append(nn.AvgPool2d(3, stride=2, padding=1))
        if inplanes != planes*block.expansion:
            downsample.append(nn.Conv2d(inplanes, planes*block.expansion, kernel_size=1, stride=1, bias=False))
            downsample.append(nn.BatchNorm2d(planes*block.expansion))
        # print(downsample)
        downsample = None if downsample == [] else nn.Sequential(*downsample)
        layers = []
        if is_fuse:
            layers.append(block(inplanes=inplanes, planes=planes, 
                                cardinality=self.cardinality, base_width=self.base_width, 
                                stride=stride, downsample=downsample,
                                patch_groups=self.patch_groups, base_scale=base_scale, is_first = False))
        else:
            layers.append(block(inplanes=inplanes, planes=planes, stride=stride, downsample=downsample,
                                cardinality=self.cardinality, base_width=self.base_width, 
                                patch_groups=self.patch_groups,base_scale=base_scale, is_first = True))
            inplanes = planes * 4
            for i in range(1, blocks):
                layers.append(block(inplanes=inplanes, planes=planes, 
                                cardinality=self.cardinality, base_width=self.base_width,
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
            x_refine = self.refine_transform(x_refine)
        
        if self.fuse_se:
            x_base = self.se_base(x_base)
            x_refine = self.se_refine(x_refine)
        _,_,h,w = x_refine.shape
        x_base = F.interpolate(x_base, size = (h,w))
        out = self.relu(x_base + x_refine)
        # print(out.shape)
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
        _,_,h,w = x_refine.shape
        if self.alpha > 1:
            x_base = self.base_transform(x_base)
            flops += c * x_base.shape[1] * x_base.shape[2] * x_base.shape[3]
        if self.beta > 1:
            c_in = x_refine.shape[1]
            x_refine = self.refine_transform(x_refine)
            flops += c_in * x_refine.shape[1] * x_refine.shape[2] * x_refine.shape[3]
        
        if self.fuse_se:
            x_base, _flops = self.se_base.forward_calc_flops(x_base)
            flops += _flops
            x_refine, _flops = self.se_refine.forward_calc_flops(x_refine)
            flops += _flops
        x_base = F.interpolate(x_base, size = (h,w))
        out = self.relu(x_base + x_refine)
        # print(out.shape)
        out, _flops = self.fusion[0].forward_calc_flops(out)
        flops += _flops
        return out, _masks, flops

class sarResNeXt(nn.Module):
    def __init__(self, block_base, block_refine, layers, cardinality, base_width,
                num_classes=10, patch_groups=1, mask_size=4, width=1, alpha=1,beta=1, base_scale=2, fuse_se=False):
        # print(num_channels)
        self.inplanes = 64
        super(sarResNeXt, self).__init__()
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.cardinality = cardinality
        self.base_width = base_width
        self.layer1 = sarModule(block_base, block_refine, 64, 64, layers[0], 
                    cardinality=self.cardinality, base_width=self.base_width, 
                    stride=2, groups=patch_groups, mask_size=mask_size, alpha=alpha,beta=beta, base_scale=base_scale, fuse_se=fuse_se)
        self.layer2 = sarModule(block_base, block_refine, 64*4, 128, layers[1], 
                                cardinality=self.cardinality, base_width=self.base_width,
                                stride=2, groups=patch_groups, mask_size=mask_size, alpha=alpha,beta=beta, base_scale=base_scale, fuse_se=fuse_se)
        self.layer3 = sarModule(block_base, block_refine, 128*4, 256, layers[2], 
                               cardinality=self.cardinality, base_width=self.base_width,
                               stride=1, groups=patch_groups, mask_size=2, alpha=alpha,beta=beta, base_scale=2, fuse_se=fuse_se)
        self.gappool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256*4, num_classes)

        for k, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # if 'gs' in str(k):
                #     m.weight.data.normal_(0, 0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each block.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, ResNeXtBottleneck) or isinstance(m, ResNeXtBottleneck_refine):
                nn.init.constant_(m.bn3.weight, 0)

        self.mask_param_num = 0
        self.backbone_param_num = 0
        for name, params in self.named_parameters():
            if 'gs' in name:
                self.mask_param_num += params.numel()
            else:
                self.backbone_param_num += params.numel()

        # assert(0==1)


    def get_mask_params(self):
        for name, params in self.named_parameters():
            if 'gs' in name:
                yield params

    def get_backbone_params(self):
        for name, params in self.named_parameters():
            if 'gs' not in name:
                yield params


    def forward(self, x, temperature=1.0, inference=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        _masks = []
        x, mask = self.layer1(x, temperature=temperature, inference=inference)
        _masks.extend(mask)
        x, mask = self.layer2(x, temperature=temperature, inference=inference)
        _masks.extend(mask)
        x, mask = self.layer3(x, temperature=temperature, inference=inference)
        _masks.extend(mask)

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
        # print('before stage 1 ', x.shape)
        x, mask, _flops = self.layer1.forward_calc_flops(x, temperature=temperature, inference=inference)
        _masks.extend(mask)
        flops += _flops
        
        # print('before stage 2 ', x.shape)
        x, mask, _flops = self.layer2.forward_calc_flops(x, temperature=temperature, inference=inference)
        _masks.extend(mask)
        flops += _flops

        # print('before stage 3 ', x.shape)
        x, mask, _flops = self.layer3.forward_calc_flops(x, temperature=temperature, inference=inference)
        _masks.extend(mask)
        flops += _flops

        flops += x.shape[1] * x.shape[2] * x.shape[3]
        x = self.gappool(x)
        x = x.view(x.size(0), -1)
        c_in = x.shape[1]
        x = self.fc(x)
        flops += c_in * x.shape[1]

        return x, _masks, flops


def sar_resneXt_cifar(depth, num_classes=100, patch_groups=2, mask_size=4, 
                    cardinality=8, base_width=64,
                    width=1, alpha=1,beta=1, base_scale=2, fuse_se=False):
    layers = {
        29: [3,3,3]
    }[depth]
    model = sarResNeXt(block_base=ResNeXtBottleneck, block_refine=ResNeXtBottleneck_refine, 
                    layers=layers, cardinality=cardinality, base_width=base_width,
                    num_classes=num_classes, patch_groups=patch_groups, 
                    mask_size=mask_size, width=width, alpha=alpha, beta=beta, base_scale=base_scale, fuse_se=fuse_se)
    return model


def sar_resneXt8x64_cifar(args):
    return sar_resneXt_cifar(depth=29, cardinality=8, base_width=64, 
                num_classes=args.num_classes, patch_groups=args.patch_groups, 
                mask_size=args.mask_size, width=2, alpha=args.alpha, beta=args.beta, 
                base_scale=args.base_scale, 
                fuse_se=False)

def sar_resneXt8x32_cifar(args):
    return sar_resneXt_cifar(depth=29, cardinality=8, base_width=32, 
                num_classes=args.num_classes, patch_groups=args.patch_groups, 
                mask_size=args.mask_size, width=2, alpha=args.alpha, beta=args.beta, 
                base_scale=args.base_scale, 
                fuse_se=False)

if __name__ == "__main__":
    import argparse
    from op_counter import measure_model
    import numpy as np
    def params_count(model):
        return np.sum([p.numel() for p in model.parameters()]).item()
    parser = argparse.ArgumentParser(description='PyTorch SARNet')
    args = parser.parse_args()
    args.num_classes = 10
    args.patch_groups = 1
    args.mask_size = 4
    args.alpha = 1
    args.beta = 1
    args.base_scale = 2
    # with torch.no_grad():
    sar_res = sar_resneXt8x32_cifar(args)
    print(sar_res)
    x = torch.rand(1,3,32,32)
    sar_res.eval()
    y1,_, flops = sar_res.forward_calc_flops(x,inference=False,temperature=1e-8)
    print(flops / 1e9)