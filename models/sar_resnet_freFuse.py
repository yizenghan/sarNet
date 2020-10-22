import torch.nn as nn
# from torch.hub import load_state_dict_from_url
import torch
import torch.nn.functional as F
from .gumbel_softmax import GumbleSoftmax
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

__all__ = ['sar_resnet_freFuse']

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
                 do_patch=False, patch_groups=1, base_scale=2, mask_size=7, alpha=1):
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

            self.conv1_refine = conv3x3(inplanes, planes, stride,groups=patch_groups)
            self.bn1_refine = norm_layer(planes)

            self.conv2_refine = conv3x3(planes, planes, stride,groups=patch_groups)
            self.bn2_refine = norm_layer(planes)
            self.flops_per_pixel_refine1 = inplanes * planes * 9 / patch_groups
            self.flops_per_pixel_refine2 = planes * planes * 9 / patch_groups

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
            flops += self.flops_per_pixel_1 * out.shape[2] * out.shape[3]

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
            # ratio = 0.3
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

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, 
                 do_patch=False, patch_groups=1, base_scale=2, mask_size=7, alpha=1):
        super(Bottleneck, self).__init__()
        self.do_patch = do_patch
        # print(do_patch, mask_size)
        if not do_patch:
            self.conv1 = nn.Conv2d(inplanes, planes // self.expansion, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes // self.expansion)
            self.conv2 = nn.Conv2d(planes // self.expansion, planes // self.expansion, kernel_size=3, stride=stride,
                                padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes // self.expansion)
            self.conv3 = nn.Conv2d(planes // self.expansion, planes, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes)
            self.flops_per_pixel_1 = inplanes * planes // self.expansion
            self.flops_per_pixel_2 = (planes // self.expansion)**2 * 9
            self.flops_per_pixel_3 = planes * planes // self.expansion
        else:
            self.mask_gen = maskGen(groups=patch_groups,inplanes=inplanes,mask_size=mask_size)
            self.first_downsample = nn.AvgPool2d(3, stride=2, padding=1)
            self.second_downsample = nn.AvgPool2d(3, stride=2, padding=1) if base_scale==4 else None

            self.conv1_base = nn.Conv2d(inplanes, planes // self.expansion // alpha, kernel_size=1, bias=False)
            self.bn1_base = nn.BatchNorm2d(planes // self.expansion // alpha)
            self.conv2_base = nn.Conv2d(planes // self.expansion // alpha, planes // self.expansion // alpha, kernel_size=3, stride=stride,
                                padding=1, bias=False)
            self.bn2_base = nn.BatchNorm2d(planes // self.expansion // alpha)
            self.conv3_base = nn.Conv2d(planes // self.expansion // alpha, planes, kernel_size=1, bias=False)
            self.bn3_base = nn.BatchNorm2d(planes)

            self.flops_per_pixel_base1 = inplanes * planes // self.expansion // alpha
            self.flops_per_pixel_base2 = (planes // self.expansion // alpha)**2 * 9
            self.flops_per_pixel_base3 = planes * planes // self.expansion // alpha

            self.conv1_refine = nn.Conv2d(inplanes, planes // self.expansion, kernel_size=1, bias=False, groups=patch_groups)
            self.bn1_refine = nn.BatchNorm2d(planes // self.expansion)
            self.conv2_refine = nn.Conv2d(planes // self.expansion, planes // self.expansion, kernel_size=3, stride=stride,
                                padding=1, bias=False, groups=patch_groups)
            self.bn2_refine = nn.BatchNorm2d(planes // self.expansion)
            self.conv3_refine = nn.Conv2d(planes // self.expansion, planes, kernel_size=1, bias=False, groups=patch_groups)
            self.bn3_refine = nn.BatchNorm2d(planes)

            self.flops_per_pixel_refine1 = inplanes * planes // self.expansion / patch_groups
            self.flops_per_pixel_refine2 = (planes // self.expansion)**2 * 9 / patch_groups
            self.flops_per_pixel_refine3 = planes * planes // self.expansion / patch_groups

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample      

    def forward(self, x, temperature=1, inference=False):
        if not self.do_patch:
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
            x_base = self.relu(x_base)

            x_base = self.conv3_base(x_base)
            x_base = self.bn3_base(x_base)

            b,c,h,w = x.shape
            mask = self.mask_gen(x, temperature=temperature)

            if not inference:
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
                x_refine = self.relu(x_refine)

                x_refine = x_refine * mask2
                x_refine = self.conv3_refine(x_refine)
                x_refine = self.bn3_refine(x_refine)
            else:
                if mask.sum() == 0.0:
                    x_refine = self.bn3(torch.zeros(residual.shape))
                else:
                    b,c,h,w = x.shape
                    
                    g = mask.shape[1]
                    m_h = mask.shape[2]
                    mask1 = mask.clone()
                    if g > 1:
                        mask1 = mask.unsqueeze(1).repeat(1,c//g,1,1,1).transpose(1,2).reshape(b,c,m_h,m_h)
                        
                    mask1 = F.interpolate(mask1, size = (h,w))
                    # print(mask1.shape, x.shape)
                    out = x * mask1

                    x_ = _extract_from_mask(out, mask)
                    
                    x_refine = []
                    pp = 0

                    for i in range(g):
                        c_out_g = self.conv1.out_channels // g

                        if mask[0,i,:,:].sum() == 0:
                            continue
                        weight = self.conv1.weight
                        weight_g = weight[i*c_out_g:(i+1)*c_out_g,:,:,:]          
                        
                        out = F.conv2d(x_[pp], weight_g, padding = 0)
                                    
                        rm = self.bn1.running_mean[i*c_out_g:(i+1)*c_out_g]
                        rv = self.bn1.running_var[i*c_out_g:(i+1)*c_out_g]
                        w_bn = self.bn1.weight[i*c_out_g:(i+1)*c_out_g]
                        b_bn = self.bn1.bias[i*c_out_g:(i+1)*c_out_g]
                        
                        out = F.batch_norm(out, running_mean=rm, running_var=rv, weight=w_bn, bias=b_bn, training=self.training, momentum=0.1, eps=1e-05)
                        # out = self.bn1(out)
                        out = self.relu(out)


                        weight = self.conv2.weight
                        c_out_g = self.conv2.out_channels //g
                        weight_g = weight[i*c_out_g:(i+1)*c_out_g,:,:,:]          
                        
                        out = F.conv2d(out, weight_g, padding = 0)
                        
                        rm = self.bn2.running_mean[i*c_out_g:(i+1)*c_out_g]
                        rv = self.bn2.running_var[i*c_out_g:(i+1)*c_out_g]
                        w_bn = self.bn2.weight[i*c_out_g:(i+1)*c_out_g]
                        b_bn = self.bn2.bias[i*c_out_g:(i+1)*c_out_g]
                        
                        out = F.batch_norm(out, running_mean=rm, running_var=rv, weight=w_bn, bias=b_bn, training=self.training, momentum=0.1, eps=1e-05)
                        out = self.relu(out)


                        weight = self.conv3.weight
                        c_out_g = self.conv3.out_channels //g
                        weight_g = weight[i*c_out_g:(i+1)*c_out_g,:,:,:]          
                        
                        out = F.conv2d(out, weight_g, padding = 0)
                        
                        rm = self.bn3.running_mean[i*c_out_g:(i+1)*c_out_g]
                        rv = self.bn3.running_var[i*c_out_g:(i+1)*c_out_g]
                        w_bn = self.bn3.weight[i*c_out_g:(i+1)*c_out_g]
                        b_bn = self.bn3.bias[i*c_out_g:(i+1)*c_out_g]
                        
                        out = F.batch_norm(out, running_mean=rm, running_var=rv, weight=w_bn, bias=b_bn, training=self.training, momentum=0.1, eps=1e-05)
                        x_refine.append(out)

                        pp +=1
                    
                    x_refine = _rearrange_features(x_refine, mask)

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
            out = self.bn1(out)
            out = self.relu(out)
            flops += self.flops_per_pixel_1 * out.shape[2] * out.shape[3]

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)
            flops += self.flops_per_pixel_2 * out.shape[2] * out.shape[3]

            out = self.conv3(out)
            out = self.bn3(out)
            flops += self.flops_per_pixel_3 * out.shape[2] * out.shape[3]

            if self.downsample is not None:
                c_in = x.shape[1]
                residual = self.downsample(x)
                _, c_out, h, w = residual.shape
                flops += c_in * c_out * h * w

            out += residual
            out = self.relu(out)
            return out, None, flops
        else:
            residual = x
            x_base = self.first_downsample(x)
            _, c, h, w = x_base.shape
            flops += c * h * w * 9
            if self.second_downsample is not None:
                x_base = self.second_downsample(x_base)
                _, c, h, w = x_base.shape
                flops += c * h * w * 9

            x_base = self.conv1_base(x_base)
            x_base = self.bn1_base(x_base)
            x_base = self.relu(x_base)
            flops += self.flops_per_pixel_base1 * x_base.shape[2] * x_base.shape[3]

            x_base = self.conv2_base(x_base)
            x_base = self.bn2_base(x_base)
            x_base = self.relu(x_base)
            flops += self.flops_per_pixel_base2 * x_base.shape[2] * x_base.shape[3]

            x_base = self.conv3_base(x_base)
            x_base = self.bn3_base(x_base)
            flops += self.flops_per_pixel_base3 * x_base.shape[2] * x_base.shape[3]
            b,c,h,w = x.shape
            mask, _flops = self.mask_gen.forward_calc_flops(x, temperature=temperature)
            ratio = mask.sum() / mask.numel()
            # ratio = 0.3
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
            x_refine = self.relu(x_refine)
            flops += ratio * self.flops_per_pixel_refine2 * x_refine.shape[2] * x_refine.shape[3]

            x_refine = x_refine * mask2
            x_refine = self.conv3_refine(x_refine)
            x_refine = self.bn3_refine(x_refine)
            flops += ratio * self.flops_per_pixel_refine3 * x_refine.shape[2] * x_refine.shape[3]

            _,_,h,w = x_refine.shape
            x_base = F.interpolate(x_base, size = (h,w))
            out = x_refine + x_base
            
            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
            out = self.relu(out)
            return out, mask, flops
        
class sarResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, patch_groups=1, mask_size1=7,mask_size2=1, width=1.0, alpha=1, base_scale=2):
        num_channels = [64,128,256, 512]
        # print(num_channels)
        self.inplanes = 64
        super(sarResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, num_channels[0], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels[0])
        self.relu = nn.ReLU(inplace=True)

        self.b_conv0 = nn.Conv2d(num_channels[0], num_channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_b0 = nn.BatchNorm2d(num_channels[0])

        # alpha = 2
        self.l_conv0 = nn.Conv2d(num_channels[0], num_channels[0] // alpha,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_l0 = nn.BatchNorm2d(num_channels[0] // alpha)
        self.l_conv1 = nn.Conv2d(num_channels[0] // alpha, num_channels[0] //
                                 alpha, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_l1 = nn.BatchNorm2d(num_channels[0] // alpha)
        self.l_conv2 = nn.Conv2d(num_channels[0] // alpha, num_channels[0], kernel_size=1, stride=1, bias=False)
        self.bn_l2 = nn.BatchNorm2d(num_channels[0])

        self.bl_init = nn.Conv2d(num_channels[0], num_channels[0], kernel_size=1, stride=1, bias=False)
        self.bn_bl_init = nn.BatchNorm2d(num_channels[0])
        self.patch_groups = patch_groups
        self.layer1 = self._make_layer(block, num_channels[0], num_channels[0]*block.expansion, 
                    layers[0], stride=1,
                    mask_size=mask_size1, alpha=alpha, base_scale=base_scale, do_patch=True)

        self.layer2 = self._make_layer(block, num_channels[0]*block.expansion,
                    num_channels[1]*block.expansion, layers[1], stride=2, 
                    mask_size=mask_size1, alpha=alpha, base_scale=base_scale, do_patch=True)
        
        self.layer3 = self._make_layer(block, num_channels[1]*block.expansion,
                    num_channels[2]*block.expansion, layers[2], stride=2, 
                    mask_size=mask_size2, alpha=alpha, base_scale=2, do_patch=True)

        self.layer4 = self._make_layer(
            block, num_channels[2]*block.expansion, num_channels[3]*block.expansion, 
            layers[3], stride=2, do_patch=False)

        self.gappool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_channels[3]*block.expansion, num_classes)

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
        for k, m in self.named_modules():
            if isinstance(m, nn.BatchNorm2d) and '3' in k:
                nn.init.constant_(m.weight, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, 
                    mask_size=7, alpha=2, base_scale=2, do_patch=True):
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
                                patch_groups = self.patch_groups, mask_size=mask_size, alpha=alpha, base_scale=base_scale, do_patch=False))
        else:
            layers.append(block(inplanes=inplanes, planes=planes, stride=stride, downsample=downsample,
                                patch_groups = self.patch_groups, mask_size=mask_size, alpha=alpha, base_scale=base_scale, do_patch=do_patch))
        for _ in range(1, blocks):
            layers.append(block(inplanes=planes, planes=planes,
                                patch_groups = self.patch_groups, mask_size=mask_size, alpha=alpha, base_scale=base_scale, do_patch=do_patch))

        return nn.ModuleList(layers)

    def forward(self, x, temperature=1.0, inference=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        bx = self.b_conv0(x)
        bx = self.bn_b0(bx)
        lx = self.l_conv0(x)
        lx = self.bn_l0(lx)
        lx = self.relu(lx)
        lx = self.l_conv1(lx)
        lx = self.bn_l1(lx)
        lx = self.relu(lx)
        lx = self.l_conv2(lx)
        lx = self.bn_l2(lx)
        x = self.relu(bx + lx)
        x = self.bl_init(x)
        x = self.bn_bl_init(x)
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
        
        for i in range(len(self.layer4)):
            x, _ = self.layer4[i](x)

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

        c_in = x.shape[1]
        bx = self.b_conv0(x)
        flops += c_in * bx.shape[1] * bx.shape[2] * bx.shape[3] * self.b_conv0.weight.shape[2]*self.b_conv0.weight.shape[3]

        bx = self.bn_b0(bx)
        lx = self.l_conv0(x)
        flops += c_in * lx.shape[1] * lx.shape[2] * lx.shape[3] * self.l_conv0.weight.shape[2]*self.l_conv0.weight.shape[3]
        lx = self.bn_l0(lx)
        lx = self.relu(lx)

        c_in = lx.shape[1]
        lx = self.l_conv1(lx)
        flops += c_in * lx.shape[1] * lx.shape[2] * lx.shape[3] * self.l_conv1.weight.shape[2]*self.l_conv1.weight.shape[3]

        lx = self.bn_l1(lx)
        lx = self.relu(lx)

        c_in = lx.shape[1]
        lx = self.l_conv2(lx)
        flops += c_in * lx.shape[1] * lx.shape[2] * lx.shape[3] * self.l_conv2.weight.shape[2]*self.l_conv2.weight.shape[3]
        lx = self.bn_l2(lx)
        x = self.relu(bx + lx)

        c_in = x.shape[1]
        x = self.bl_init(x)
        flops += c_in * x.shape[1] * x.shape[2] * x.shape[3] * self.bl_init.weight.shape[2]*self.bl_init.weight.shape[3]
        x = self.bn_bl_init(x)
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

        for i in range(len(self.layer4)):
            x,_, _flops = self.layer4[i].forward_calc_flops(x)
            flops += _flops

        flops += x.shape[1] * x.shape[2] * x.shape[3]
        x = self.gappool(x)
        x = x.view(x.size(0), -1)
        c_in = x.shape[1]
        x = self.fc(x)
        flops += c_in * x.shape[1]

        return x, _masks, flops

def sar_resnet_freFuse(depth, num_classes=1000, patch_groups=1, mask_size1=7, mask_size2=2, width=1.0, alpha=1, base_scale=2):
    layers = {
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [4, 8, 18, 3],
        152: [5, 12, 30, 3]
    }[depth]
    block = BasicBlock if depth == 34 else Bottleneck
    model = sarResNet(block=block, layers=layers, 
                      num_classes=num_classes, patch_groups=patch_groups, mask_size1=mask_size1, mask_size2=mask_size2, 
                      width=width, alpha=alpha, base_scale=base_scale)
    return model


def _extract_from_mask(x, mask):
    pad = nn.ConstantPad2d(padding=(1, 1, 1, 1), value=0.0)
    feats = []

    b,c_f,h_f,w_f = x.size()
    _,g,h,w = mask.size()
      
    x = pad(x)    

    h_interval = h_f // h
    w_interval = w_f // w
    c_interval = c_f // g

    for k in range(g):
        if mask[0,k,:,:].sum() == 0:
            continue
        feat=[]        
        for i in range(h):
            h_1 = i*h_interval
            for j in range(w):
                w_1 = j*w_interval
                idx = mask[0, k, i, j]
                if idx > 0:
                    tp = x[0, k*c_interval:(k+1)*c_interval, h_1:h_1+h_interval+2, w_1:w_1+w_interval+2]
                    feat.append(tp)
        
        feat = torch.stack(feat)
        feats.append(feat)
    return feats

def _rearrange_features(feat, mask):
    b,c_f,h_f,w_f = feat[0].size()
    _,c,h,w = mask.size()
    
    h_interval = h_f
    w_interval = w_f
    c_interval = c_f
    x = torch.zeros(1, c*c_f, h*h_f, w*w_f)

    q = 0

    pp = 0
    for k in range(c):
        if mask[0,k,:,:].sum() == 0:
            continue
        for i in range(h):
            h_1 = i*h_interval 
            for j in range(w):
                w_1 = j*w_interval
                idx = mask[0, k, i, j]
                if idx > 0:
                    x[:,k*c_interval:(k+1)*c_interval, h_1:h_1+h_interval, w_1:w_1+w_interval] = feat[pp][q,:,:,:]
                    q += 1
        if q >= mask[0,k,:,:].sum():
            q = 0
        pp += 1
    # print(x)
    return x


if __name__ == "__main__":
    
    from op_counter import measure_model
    
    # print(sar_res)
    sar_res = sar_resnet_freFuse(depth=50, patch_groups=4, width=1, alpha=4, base_scale=2)
    # with torch.no_grad():
        
    # print(sar_res)
    x = torch.rand(1,3,224,224)
    sar_res.eval()
    # y,_mask = sar_res(x,inference=False,temperature=1e-8)
    y, _masks, flops = sar_res.forward_calc_flops(x,inference=False,temperature=1e-8)
    print(flops / 1e9)