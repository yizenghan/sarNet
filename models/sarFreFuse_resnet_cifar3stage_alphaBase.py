import torch.nn as nn
# from torch.hub import load_state_dict_from_url
import torch
import torch.nn.functional as F
from gumbel_softmax import GumbleSoftmax
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


class maskGen(nn.Module):
    def __init__(self, groups=1, inplanes=16, mask_size=7):
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
        self.fc_gs.bias.data[:2*groups:2] = 0.1
        self.fc_gs.bias.data[1:2*groups+1:2] = 5.0      
        self.gs = GumbleSoftmax()

    def forward(self, x, temperature=1.0):
        gates = self.conv3x3_gs(x)
        gates = self.pool(gates)
        gates = self.fc_gs(gates)
        gates = gates.view(x.shape[0],self.groups,2,self.mask_size,self.mask_size)
        gates = self.gs(gates, temp=temperature, force_hard=True)
        gates = gates[:,:,1,:,:]
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
        gates = gates.view(x.shape[0],self.groups,2,self.mask_size,self.mask_size)
        # print(temperature)
        gates = self.gs(gates, temp=temperature, force_hard=True)
        gates = gates[:,:,1,:,:]
        return gates, flops


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1,groups=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,groups=groups)

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 do_patch=False, patch_groups=1, base_scale=2, mask_size=7, alpha=1, beta=1):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        # print(do_patch)
        self.do_patch = do_patch
        self.relu = nn.ReLU(inplace=True)

        if not do_patch:
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.bn1 = norm_layer(planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = conv3x3(planes, planes)
            self.bn2 = norm_layer(planes)
            self.flops_per_pixel_1 = inplanes * planes * 9
            self.flops_per_pixel_2 = planes * planes * 9
        else:
            self.mask_gen = maskGen(groups=patch_groups,inplanes=inplanes,mask_size=mask_size)
            self.first_downsample = nn.AvgPool2d(3, stride=2, padding=1)
            self.second_downsample = nn.AvgPool2d(3, stride=2, padding=1) if base_scale==4 else None
            self.conv1_base = conv3x3(inplanes, planes // alpha, stride)
            self.bn1_base = norm_layer(planes // alpha)
            self.conv2_base = conv3x3(planes // alpha, planes)
            self.bn2_base = norm_layer(planes)

            self.flops_per_pixel_base1 = inplanes * planes * 9 // alpha
            self.flops_per_pixel_base2 = planes * planes * 9 // alpha

            self.conv1_refine = conv3x3(inplanes, planes*beta, stride,groups=patch_groups)
            self.bn1_refine = norm_layer(planes*beta)

            self.conv2_refine = conv3x3(planes*beta, planes, stride,groups=patch_groups)
            self.bn2_refine = norm_layer(planes)
            self.flops_per_pixel_refine1 = inplanes * planes*beta * 9 / patch_groups
            self.flops_per_pixel_refine2 = planes*beta * planes * 9 / patch_groups

        self.downsample = downsample

    def forward(self, x, temperature=1.0, inference=False):
        if not self.do_patch:
            residual = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
            out = self.relu(out)

            return out, None
        else:
            residual = x

            x_base = self.first_downsample(x)
            if self.second_downsample is not None:
                x_base = self.second_downsample(x_base)
            x_base = self.conv1_base(x_base)
            x_base = self.bn1_base(x_base)
            x_base = self.relu(x_base)

            x_base = self.conv2_base(x_base)
            x_base = self.bn2_base(x_base)

            b,c,h,w = x.shape
            mask = self.mask_gen(x, temperature=temperature)
            g = mask.shape[1]
            m_h = mask.shape[2]
            mask1 = mask.clone()
            if g > 1:
                mask1 = mask1.unsqueeze(1).repeat(1,c//g,1,1,1).transpose(1,2).reshape(b,c,m_h,m_h)
                
            mask1 = F.interpolate(mask1, size = (h,w))
            x_refine = x * mask1
            x_refine = self.conv1_refine(x_refine)
            x_refine = self.bn1_refine(x_refine)
            x_refine = self.relu(x_refine)

            mask2 = mask
            _,c,h,w = x_refine.shape
            if g > 1:
                mask2 = mask2.unsqueeze(1).repeat(1,c//g,1,1,1).transpose(1,2).reshape(b,c,m_h,m_h)
            mask2 = F.interpolate(mask2, size = (h,w))
            x_refine = x_refine * mask2
            x_refine = self.conv2_refine(x_refine)
            x_refine = self.bn2_refine(x_refine)

            _,_,h,w = x_refine.shape
            x_base = F.interpolate(x_base, size = (h,w))
            out = x_refine + x_base
            
            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
            out = self.relu(out)
            return out, mask

    def forward_calc_flops(self, x, temperature=1.0, inference=False):
        flops = 0
        if not self.do_patch:
            residual = x
            out = self.conv1(x)
            flops += self.flops_per_pixel_1 * out.shape[2] * out.shape[3]
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            flops += self.flops_per_pixel_2 * out.shape[2] * out.shape[3]

            if self.downsample is not None:
                c_in = x.shape[1]
                residual = self.downsample(x)
                _,c_out,h,w = residual.shape
                flops += c_in * c_out * h * w

            out += residual
            out = self.relu(out)

            return out, None, flops
        else:
            residual = x

            x_base = self.first_downsample(x)
            flops += 9 * x_base.shape[1] * x_base.shape[2] * x_base.shape[3]
            if self.second_downsample is not None:
                x_base = self.second_downsample(x_base)
                flops += 9 * x_base.shape[1] * x_base.shape[2] * x_base.shape[3]
            x_base = self.conv1_base(x_base)
            x_base = self.bn1_base(x_base)
            x_base = self.relu(x_base)
            flops += self.flops_per_pixel_base1 * x_base.shape[2] * x_base.shape[3]

            x_base = self.conv2_base(x_base)
            x_base = self.bn2_base(x_base)
            flops += self.flops_per_pixel_base2 * x_base.shape[2] * x_base.shape[3]

            b,c,h,w = x.shape
            mask, _flops = self.mask_gen.forward_calc_flops(x, temperature=temperature)
            ratio = mask.sum() / mask.numel()
            # print(ratio)
            # ratio = 0.7564
            flops += _flops
            g = mask.shape[1]
            m_h = mask.shape[2]
            mask1 = mask.clone()
            if g > 1:
                mask1 = mask1.unsqueeze(1).repeat(1,c//g,1,1,1).transpose(1,2).reshape(b,c,m_h,m_h)
                
            mask1 = F.interpolate(mask1, size = (h,w))
            x_refine = x * mask1
            x_refine = self.conv1_refine(x_refine)
            x_refine = self.bn1_refine(x_refine)
            x_refine = self.relu(x_refine)
            flops += ratio * self.flops_per_pixel_refine1 * x_refine.shape[2] * x_refine.shape[3]

            mask2 = mask
            _,c,h,w = x_refine.shape
            if g > 1:
                mask2 = mask2.unsqueeze(1).repeat(1,c//g,1,1,1).transpose(1,2).reshape(b,c,m_h,m_h)
            mask2 = F.interpolate(mask2, size = (h,w))
            x_refine = x_refine * mask2
            x_refine = self.conv2_refine(x_refine)
            x_refine = self.bn2_refine(x_refine)
            flops += ratio * self.flops_per_pixel_refine2 * x_refine.shape[2] * x_refine.shape[3]

            _,_,h,w = x_refine.shape
            x_base = F.interpolate(x_base, size = (h,w))
            out = x_refine + x_base
            
            if self.downsample is not None:
                c_in = x.shape[1]
                residual = self.downsample(x)
                flops += c_in * residual.shape[1] *residual.shape[2] *residual.shape[3]  

            out += residual
            out = self.relu(out)
            return out, mask, flops

        
class sarResNet(nn.Module):
    def __init__(self, block, layers, num_classes=100, patch_groups=1, mask_size1=4, mask_size2=1, width=1, alpha=1,beta=1, base_scale=2):
        # print(num_channels)
        self.inplanes = 16
        super(sarResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.patch_groups = patch_groups
        self.layer1 = self._make_layer(block, self.inplanes, 16*width, layers[0], stride=1,
                    mask_size=mask_size1, alpha=alpha,beta=beta, base_scale=base_scale, do_patch=True)

        self.layer2 = self._make_layer(block, 16*width, 32*width, layers[1], stride=2,
                    mask_size=mask_size1, alpha=alpha,beta=beta, base_scale=base_scale, do_patch=True)
        
        self.layer3 = self._make_layer(block, 32*width, 64*width, layers[2], stride=2,
                    mask_size=mask_size2, alpha=alpha,beta=beta, base_scale=2, do_patch=True)

        self.gappool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64*width, num_classes)

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
        # for k, m in self.named_modules():
        #     if isinstance(m, nn.BatchNorm2d) and 'bn2' in k:
        #         nn.init.constant_(m.weight, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, 
                    mask_size=7, alpha=2,beta=1, base_scale=2, do_patch=True):
        downsample = []
        if stride != 1:
            downsample.append(nn.AvgPool2d(3, stride=2, padding=1))
        if inplanes != planes:
            downsample.append(nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False))
            downsample.append(nn.BatchNorm2d(planes))
        downsample = None if downsample == [] else nn.Sequential(*downsample)

        layers = []
        if stride != 1:
            layers.append(block(inplanes=inplanes, planes=planes, stride=stride, downsample=downsample,
                                patch_groups = self.patch_groups, mask_size=mask_size, alpha=alpha,beta=beta, base_scale=base_scale, do_patch=False))
        else:
            layers.append(block(inplanes=inplanes, planes=planes, stride=stride, downsample=downsample,
                                patch_groups = self.patch_groups, mask_size=mask_size, alpha=alpha,beta=beta, base_scale=base_scale, do_patch=do_patch))
        for _ in range(1, blocks):
            layers.append(block(inplanes=planes, planes=planes,
                                patch_groups = self.patch_groups, mask_size=mask_size, alpha=alpha,beta=beta, base_scale=base_scale, do_patch=do_patch))

        return nn.ModuleList(layers)

    def forward(self, x, temperature=1.0, inference=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # print('before layer 1:', x.shape)
        _masks = []
        for i in range(len(self.layer1)):
            x, mask = self.layer1[i](x, temperature=temperature, inference=inference)
            _masks.extend(mask)
        
        x, _ = self.layer2[0](x)
        for i in range(1, len(self.layer2)):
            x, mask = self.layer2[i](x, temperature=temperature, inference=inference)
            _masks.extend(mask)
       
        x, _ = self.layer3[0](x)
        for i in range(1, len(self.layer3)):
            x, mask = self.layer3[i](x, temperature=temperature, inference=inference)
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
        for i in range(len(self.layer1)):
            x, mask, _flops = self.layer1[i].forward_calc_flops(x, temperature=temperature, inference=inference)
            flops += _flops
            _masks.extend(mask)
        
        x,_,_flops = self.layer2[0].forward_calc_flops(x)
        flops += _flops
        for i in range(1, len(self.layer2)):
            x, mask, _flops = self.layer2[i].forward_calc_flops(x, temperature=temperature, inference=inference)
            flops += _flops
            _masks.extend(mask)
       
        x,_,_flops = self.layer3[0].forward_calc_flops(x)
        flops += _flops
        for i in range(1, len(self.layer3)):
            x, mask, _flops = self.layer3[i].forward_calc_flops(x, temperature=temperature, inference=inference)
            flops += _flops
            _masks.extend(mask)

        flops += x.shape[1] * x.shape[2] * x.shape[3]
        x = self.gappool(x)
        x = x.view(x.size(0), -1)
        c_in = x.shape[1]
        x = self.fc(x)
        flops += c_in * x.shape[1]

        return x, _masks, flops

def sar_resnet_freFuse_cifar(depth, num_classes=100, patch_groups=2, mask_size1=4, mask_size2=1, width=1, alpha=1, beta=1, base_scale=2):
    layers = {
        16: [2, 2, 2],
        20: [3, 3, 3],
        32: [5, 5, 5],
        56: [9, 9, 9],
        110: [18, 18, 18],
        164: [27, 27, 27]
    }[depth]

    model = sarResNet(block=BasicBlock, layers=layers,
                      num_classes=num_classes, patch_groups=patch_groups, mask_size1=mask_size1, mask_size2=mask_size2, 
                      width=width, alpha=alpha, beta=beta, base_scale=base_scale)
    return model


def sarFrefuse_resnet32x2_alphaBase_cifar(args):
    return sar_resnet_freFuse_cifar(depth=32, width=2, num_classes=args.num_classes, patch_groups=args.patch_groups, mask_size1=args.mask_size,mask_size2=2, 
                                    alpha=args.alpha, beta=args.beta, base_scale=args.base_scale)

if __name__ == "__main__":
    
    import argparse
    from op_counter import measure_model
    parser = argparse.ArgumentParser(description='PyTorch SARNet')
    args = parser.parse_args()
    args.num_classes = 10
    args.patch_groups = 2
    args.mask_size = 4
    args.alpha = 2
    args.beta = 1
    args.base_scale = 2
    with torch.no_grad():
        sar_res = sarFrefuse_resnet32x2_alphaBase_cifar(args)
        print(sar_res)
        sar_res.eval()
        x = torch.randn(1,3,32,32)
        y1, _masks, flops = sar_res.forward_calc_flops(x,inference=False,temperature=1e-8)
        print(len(_masks))
        # print(_masks[0])
        # print(_masks[9].shape)
        print(flops / 1e8)
        # y1 = sar_res(x,inference=True)
        # print((y-y1).abs().sum())