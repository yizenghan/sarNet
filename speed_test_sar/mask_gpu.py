import torch.nn as nn
# from torch.hub import load_state_dict_from_url
import torch
import torch.nn.functional as F
from gumbel_softmax import GumbleSoftmax
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from conv_bn_fuse import fuse_module
import argparse
    # from op_counter import measure_model
import time
import numpy as np
# torch.set_num_threads(1)
parser = argparse.ArgumentParser(description='PyTorch SARNet')
args = parser.parse_args()
args.num_classes = 1000
args.patch_groups = 2
args.mask_size = 7
args.alpha = 2
args.beta = 1
args.base_scale = 2
time_calc = []
time_extract = []
time_rearrange = []
time_conv = []
time_forward = []
time_base_branch = []
time_mask_gen = []
time_transform_fusion = []
time_stem = []
time_tail = []
time_layers = []
time_mask_define = []
time_prepare_mask = []
time_ds = []
mask14 = torch.zeros((1, args.patch_groups, 14, 14)).cuda(0)
if args.patch_groups==2:
    mask14[:,0,::2,::2] = 1.0
    mask14[:,0,1::2,1::2] = 1.0
    mask14[:,1,::2,::2] = 1.0
    mask14[:,1,1::2,1::2] = 1.0
elif args.patch_groups==1:
    mask14[:,0,::2,::2] = 1.0
    mask14[:,0,1::2,1::2] = 1.0
elif args.patch_groups==4:
    mask14[:,0,::2,::2] = 1.0
    mask14[:,0,1::2,1::2] = 1.0
    mask14[:,1,::2,::2] = 1.0
    mask14[:,1,1::2,1::2] = 1.0
    mask14[:,2,::2,::2] = 1.0
    mask14[:,2,1::2,1::2] = 1.0
    mask14[:,3,::2,::2] = 1.0
    mask14[:,3,1::2,1::2] = 1.0

mask7 = torch.zeros((1, args.patch_groups, 7, 7)).cuda(0)
if args.patch_groups==2:
    mask7[:,0,::2,::2] = 1.0
    mask7[:,0,1::2,1::2] = 1.0
    mask7[:,1,::2,::2] = 1.0
    mask7[:,1,1::2,1::2] = 1.0
elif args.patch_groups==1:
    mask7[:,0,::2,::2] = 1.0
    mask7[:,0,1::2,1::2] = 1.0
elif args.patch_groups==4:
    mask7[:,0,::2,::2] = 1.0
    mask7[:,0,1::2,1::2] = 1.0
    mask7[:,1,::2,::2] = 1.0
    mask7[:,1,1::2,1::2] = 1.0
    mask7[:,2,::2,::2] = 1.0
    mask7[:,2,1::2,1::2] = 1.0
    mask7[:,3,::2,::2] = 1.0
    mask7[:,3,1::2,1::2] = 1.0
    
mask2 = torch.zeros((1, args.patch_groups, 2, 2)).cuda(0)
if args.patch_groups==2:
    mask2[:,0,::2,::2] = 1.0
    mask2[:,0,1::2,1::2] = 1.0
    mask2[:,1,::2,::2] = 1.0
    mask2[:,1,1::2,1::2] = 1.0
elif args.patch_groups==1:
    mask2[:,0,::2,::2] = 1.0
    mask2[:,0,1::2,1::2] = 1.0
elif args.patch_groups==4:
    mask2[:,0,::2,::2] = 1.0
    mask2[:,0,1::2,1::2] = 1.0
    mask2[:,1,::2,::2] = 1.0
    mask2[:,1,1::2,1::2] = 1.0
    mask2[:,2,::2,::2] = 1.0
    mask2[:,2,1::2,1::2] = 1.0
    mask2[:,3,::2,::2] = 1.0
    mask2[:,3,1::2,1::2] = 1.0

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, last_relu=True, patch_groups=1, base_scale=2,
                 is_first=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
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
        residual = x
        c_in = x.shape[1]
        out = self.conv1(x)
        _, c_out, h, w = out.shape
        flops += c_in * c_out * h * w * 9

        out = self.bn1(out)
        out = self.relu(out)

        c_in = c_out
        out = self.conv2(out)
        _, c_out, h, w = out.shape
        flops += c_in * c_out * h * w * 9
        out = self.bn2(out)

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


class BasicBlock_refine(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, last_relu=True, patch_groups=1, base_scale=2,
                 is_first=True):
        super(BasicBlock_refine, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=patch_groups)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=patch_groups)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

        self.stride = stride
        self.last_relu = last_relu
        self.patch_groups = patch_groups
        # print(patch_groups)

    def forward(self, x, mask, inference=False):
        residual = x
        if self.downsample is not None:  # skip connection before mask
            residual = self.downsample(x)

        b, c, h, w = x.shape
        g = mask.shape[1]
        m_h = mask.shape[2]
        mask1 = mask.clone()
        if g > 1:
            mask1 = mask1.unsqueeze(1).repeat(1, c // g, 1, 1, 1).transpose(1, 2).reshape(b, c, m_h, m_h)

        mask1 = F.interpolate(mask1, size=(h, w))
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
            mask2 = mask2.unsqueeze(1).repeat(1, c_out // g, 1, 1, 1).transpose(1, 2).reshape(b, c_out, m_h, m_h)
        mask2 = F.interpolate(mask2, size=(h, w))
        # print(mask2.shape, out.shape)
        out = out * mask2
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        if self.last_relu:
            out = self.relu(out)
        return out

    def forward_calc_flops(self, x, mask, inference=False):
        # print('refine bottleneck, input shape: ', x.shape)
        residual = x
        flops = 0

        if self.downsample is not None:  # skip connection before mask
            c_in = x.shape[1]
            residual = self.downsample(x)
            flops += c_in * residual.shape[1] * residual.shape[2] * residual.shape[3]

        b, c, h, w = x.shape
        g = mask.shape[1]
        m_h = mask.shape[2]
        ratio = mask.sum() / mask.numel()
        # ratio = 0.75
        # print(ratio)
        mask1 = mask.clone()
        if g > 1:
            mask1 = mask1.unsqueeze(1).repeat(1, c // g, 1, 1, 1).transpose(1, 2).reshape(b, c, m_h, m_h)

        mask1 = F.interpolate(mask1, size=(h, w))
        # print(mask1.shape, x.shape)
        out = x * mask1
        c_in = out.shape[1]
        out = self.conv1(out)
        flops += ratio * c_in * out.shape[1] * out.shape[2] * out.shape[3] * 9 / self.conv1.groups
        out = self.bn1(out)
        out = self.relu(out)

        c_out = out.shape[1]
        # print(mask1.shape, mask.shape)
        mask2 = mask.clone()
        if g > 1:
            mask2 = mask2.unsqueeze(1).repeat(1, c_out // g, 1, 1, 1).transpose(1, 2).reshape(b, c_out, m_h, m_h)
        mask2 = F.interpolate(mask2, size=(h, w))
        out = out * mask2
        c_in = out.shape[1]
        out = self.conv2(out)
        flops += ratio * c_in * out.shape[1] * out.shape[2] * out.shape[3] * 9 / self.conv2.groups
        out = self.bn2(out)

        out += residual
        if self.last_relu:
            out = self.relu(out)
        return out, flops


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, last_relu=True, patch_groups=1,
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
        # t0_base = time.time()
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
        # t1_base = time.time()
        # time_base_branch.append((t1_base-t0_base) * 1000)
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
        _, c_out, h, w = out.shape
        flops += c_in * c_out * h * w / self.conv1.groups

        out = self.bn1(out)
        out = self.relu(out)

        c_in = c_out
        out = self.conv2(out)
        _, c_out, h, w = out.shape
        flops += c_in * c_out * h * w * 9 / self.conv2.groups

        out = self.bn2(out)
        out = self.relu(out)

        c_in = c_out
        out = self.conv3(out)
        _, c_out, h, w = out.shape
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

    def __init__(self, inplanes, planes, stride=1, downsample=None, last_relu=True, patch_groups=1, base_scale=2,
                 is_first=True):
        super(Bottleneck_refine, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes // self.expansion, kernel_size=1, bias=False, groups=patch_groups)
        self.bn1 = nn.BatchNorm2d(planes // self.expansion)
        self.conv2 = nn.Conv2d(planes // self.expansion, planes // self.expansion, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=patch_groups)
        self.bn2 = nn.BatchNorm2d(planes // self.expansion)
        self.conv3 = nn.Conv2d(planes // self.expansion, planes, kernel_size=1, bias=False, groups=patch_groups)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        self.stride = stride
        self.last_relu = last_relu
        self.patch_groups = patch_groups

    def forward(self, x, mask, inference=True):
        # t0 = time.time()
        residual = x
        if self.downsample is not None:  # skip connection before mask
            residual = self.downsample(x)
        # t1 = time.time()
        # time_ds.append((t1-t0)*1000)
        if not inference:
            b, c, h, w = x.shape
            g = mask.shape[1]
            m_h = mask.shape[2]
            if g > 1:
                mask1 = mask.unsqueeze(1).repeat(1, c // g, 1, 1, 1).transpose(1, 2).reshape(b, c, m_h, m_h)
            else:
                mask1 = mask.clone()
            mask1 = F.interpolate(mask1, size=(h, w))
            # print(mask1.shape, x.shape)
            out = x * mask1
            # print(mask1)
            out = self.conv1(out)
            out = self.bn1(out)
            out = self.relu(out)

            c_out = out.shape[1]
            # print(mask1.shape, mask.shape)
            if g > 1:
                mask2 = mask.unsqueeze(1).repeat(1, c_out // g, 1, 1, 1).transpose(1, 2).reshape(b, c_out, m_h, m_h)
            else:
                mask2 = mask.clone()
            mask2 = F.interpolate(mask2, size=(h, w))
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
        else:
            # t0 = time.time()
            if mask.sum() == 0.0:
                out = self.bn3(torch.zeros(residual.shape))
                out += residual

                if self.last_relu:
                    out = self.relu(out)
                return out

            b, c, h, w = x.shape

            g = mask.shape[1]
            mask_size = mask.shape[2]
            mask = mask.view(b,g,mask_size,mask_size,1,1,1)
            # print(x.shape, mask.shape)
            
            channels_per_group = c//g
            hw_per_patch = h // mask_size
            
            # corner0 = x[0,:channels_per_group,:hw_per_patch,:hw_per_patch]
            
            x_ = x.view(b,g,channels_per_group,mask_size,hw_per_patch,mask_size,hw_per_patch)
            # print(x_.shape)
            x_ = x_.permute(0,1,3,5,2,4,6)  # b,g,7,7,c//g,h,w
            
            # print(mask[0])
            # corner1 = x_[:,:,:,0,0,0,0]
            
            # print((corner0 - corner1).abs().sum())
            # assert(0==1)
            num_non_zero = torch.sum(mask, dim=(2,3)).squeeze()
            # if not isinstance(num_non_zero, list):
            #     num_non_zero = [num_non_zero]
            # print(num_non_zero)
            # assert(0==1)
            
            n_pixels_per_patch = channels_per_group * hw_per_patch**2
            n_pixels_per_group = n_pixels_per_patch * num_non_zero
            
            # print(n_pixels_per_group)
            # assert(0==1)
            # print(x_.shape, mask.shape)
            x_ = torch.masked_select(x_, mask>0)
            # print(x_.shape)
            
            # outs = []
            # input_x = 
            c_out_g1 = self.conv1.out_channels // g
            c_out_g2 = self.conv2.out_channels // g
            c_out_g3 = self.conv3.out_channels // g
            # inputs.append(input_x)
            # print(input_x.shape)

            out = calc_one_group(x_[:int(n_pixels_per_group[0])].view(int(num_non_zero[0]), channels_per_group, hw_per_patch, hw_per_patch),0, self.conv1, self.conv2, self.conv3,c_out_g1,c_out_g2,c_out_g3,self.relu)
            # outs.append(out)
            output = torch.zeros(b,g,mask_size,mask_size,out.shape[1],out.shape[2],out.shape[3],device=x.device)
            # print(output[0,0].shape, mask[0,0].shape, out.shape)
            output[0,0].masked_scatter_(mask[0,0]>0, out)
            # assert(0==1)
            for i in range(1, g):
                out = calc_one_group(x_[int(torch.sum(n_pixels_per_group[:i-1])):int(torch.sum(n_pixels_per_group[:i]))].view(int(num_non_zero[0]), channels_per_group, hw_per_patch, hw_per_patch),i, self.conv1, self.conv2, self.conv3,c_out_g1,c_out_g2,c_out_g3,self.relu)
                output[0,i].masked_scatter_(mask[0,i]>0, out)

   
            # t4 = time.time()
            # t_extract = t4 - t1
            # print('extract_time is:', t_extract*1000)
            # time_extract.append(t_extract*1000)

            # t4 = time.time()
            # outs = []
            # pp = 0
            
            # for i in range(g):
            #     if mask[0, i, :, :].sum() == 0:
            #         continue
            #     out = calc_one_group(inputs[i],i, self.conv1, self.conv2, self.conv3,c_out_g1,c_out_g2,c_out_g3,self.relu)
            #     outs.append(out)

            # t5 = time.time()
            # t_conv = t5 - t4
            # print('conv_time is:', t_conv*1000)
            # time_conv.append(t_conv*1000)
            # outs = _rearrange_features(outs, mask, residual)
            # assert(0==1)
            # outs = _rearrange_features(outs, mask)
            
            # b,g,7,7,c/g,h,w -> b,g,c/g, 7,h, 7,w
            # b,g,channels_per_group,mask_size,hw_per_patch,mask_size,hw_per_patch
            b,g,mask_size,_,c_per_g,hw_per_patch,_ = output.shape
            # print(output.shape)
            # output = output.permute(0,1,4,2,5,3,6).contiguous()
            # print(output.shape)
            output = output.reshape(b, g*c_per_g, mask_size*hw_per_patch, mask_size*hw_per_patch)
            output += residual
            # t6 = time.time()
            # t_rearrange = t6 - t5
            # time_rearrange.append(t_rearrange*1000)
            if self.last_relu:
                output = self.relu(output)

            # t7 = time.time()
            # t_total = t7 - t0
            # print('total_time is', t_total*1000, '--------')
            # print(t_total-t_prepare_mask-t_extract-t_conv-t_rearrange)
            # assert(0==1)
            # time_forward.append(t_total*1000)
            return output


    def forward_calc_flops(self, x, mask, inference=False):
        # print('refine bottleneck, input shape: ', x.shape)
        t1 = time.time()
        residual = x
        flops = 0
        # print('In a refine bottleneck, x shape: ', x.shape)
        if self.downsample is not None:  # skip connection before mask
            c_in = x.shape[1]
            residual = self.downsample(x)
            flops += c_in * residual.shape[1] * residual.shape[2] * residual.shape[3]

        # t1 = time.time()
        b, c, h, w = x.shape
        g = mask.shape[1]
        m_h = mask.shape[2]
        ratio = mask.sum() / mask.numel()
        # ratio = 0.7
        # print(ratio)
        mask1 = mask.clone()
        if g > 1:
            mask1 = mask1.unsqueeze(1).repeat(1, c // g, 1, 1, 1).transpose(1, 2).reshape(b, c, m_h, m_h)

        mask1 = F.interpolate(mask1, size=(h, w))
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
            mask2 = mask2.unsqueeze(1).repeat(1, c_out // g, 1, 1, 1).transpose(1, 2).reshape(b, c_out, m_h, m_h)
        mask2 = F.interpolate(mask2, size=(h, w))
        out = out * mask2
        c_in = out.shape[1]
        out = self.conv2(out)
        flops += ratio * c_in * out.shape[1] * out.shape[2] * out.shape[3] * 9 / self.conv2.groups
        out = self.bn2(out)
        out = self.relu(out)

        out = out * mask2
        c_in = out.shape[1]
        out = self.conv3(out)
        flops += ratio * c_in * out.shape[1] * out.shape[2] * out.shape[3] / self.conv3.groups
        out = self.bn3(out)
        t2=time.time()
        out += residual
        if self.last_relu:
            out = self.relu(out)
        # print('total_time is', (t2 - t1)*1000, '--------')
        time_calc.append((t2 - t1)*1000)
        return out, flops


class maskGen(nn.Module):
    def __init__(self, groups=1, inplanes=64, mask_size=7):
        super(maskGen, self).__init__()
        self.groups = groups
        self.mask_size = mask_size
        self.conv3x3_gs = nn.Sequential(
            nn.Conv2d(inplanes, groups * 4, kernel_size=3, padding=1, stride=1, bias=False, groups=groups),
            nn.BatchNorm2d(groups * 4),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool2d((mask_size, mask_size))
        self.fc_gs = nn.Conv2d(groups * 4, groups * 2, kernel_size=1, stride=1, padding=0, bias=True, groups=groups)
        self.fc_gs.bias.data[:2 * groups:2] = 5.0
        self.fc_gs.bias.data[1:2 * groups + 1:2] = 5.0
        self.gs = GumbleSoftmax()

    def forward(self, x, temperature=1.0):
        gates = self.conv3x3_gs(x)
        gates = self.pool(gates)
        gates = self.fc_gs(gates)

        # print(gates)
        # assert(0==1)
        gates = gates.view(x.shape[0], self.groups, 2, self.mask_size, self.mask_size)

        # for i in range(gates.shape[1]):
        #     print(gates[0,i,:,:,:])
        #     print('hhh')
        #
        # print(temperature)
        gates = self.gs(gates, temp=temperature, force_hard=True)
        gates = gates[:, :, 1, :, :]
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
        gates = gates.view(x.shape[0], self.groups, 2, self.mask_size, self.mask_size)
        # print(temperature)
        gates = self.gs(gates, temp=temperature, force_hard=True)
        # print('3.shape:', gates.shape, gates)
        gates = gates[:, :, 1, :, :]
        # print('4.shape:', torch.mean(gates), gates.shape, gates)
        return gates, flops


class sarModule(nn.Module):
    def __init__(self, block_base, block_refine, in_channels, out_channels, blocks, stride, groups=1, mask_size=7,
                 alpha=1, beta=1, base_scale=2):
        super(sarModule, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.patch_groups = groups
        self.mask_size = mask_size
        self.base_scale = base_scale
        assert (base_scale in [2, 4])
        self.base_scale = base_scale
        mask_gen_list = []
        for _ in range(blocks - 1):
            mask_gen_list.append(maskGen(groups=groups, inplanes=out_channels // alpha, mask_size=mask_size))
        self.mask_gen = nn.ModuleList(mask_gen_list)
        base_last_relu = True if alpha > 1 else False
        refine_last_relu = True if beta > 1 else False
        self.base_module = self._make_layer(block_base, in_channels, out_channels // alpha, blocks - 1, 2,
                                            last_relu=base_last_relu, base_scale=base_scale)

        self.refine_module = self._make_layer(block_refine, in_channels, int(out_channels * beta), blocks - 1, 1,
                                              last_relu=refine_last_relu, base_scale=base_scale)
        self.alpha = alpha
        self.beta = beta
        if alpha != 1:
            self.base_transform = nn.Sequential(
                nn.Conv2d(out_channels // alpha, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        if beta != 1:
            self.refine_transform = nn.Sequential(
                nn.Conv2d(int(out_channels * beta), out_channels, kernel_size=1, bias=False),
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
        if blocks == 1:  # fuse, is not the first of a base branch
            layers.append(block(inplanes, planes, stride=stride, downsample=downsample,
                                patch_groups=self.patch_groups, base_scale=base_scale, is_first=False))
        else:
            layers.append(block(inplanes, planes, stride, downsample, patch_groups=self.patch_groups,
                                base_scale=base_scale, is_first=True))
            for i in range(1, blocks):
                layers.append(block(planes, planes,
                                    last_relu=last_relu if i == blocks - 1 else True,
                                    patch_groups=self.patch_groups, base_scale=base_scale, is_first=False))

        return nn.ModuleList(layers)

    def forward(self, x, temperature=1e-8, inference=False):
        # t0 = time.time()
        _masks = []
        x_refine = x
        refine_ls = []
        
        for i in range(len(self.base_module)):
            x_base = self.base_module[i](x_base) if i != 0 else self.base_module[i](x)
            # t0 = time.time()
            # mask = self.mask_gen[i](x_base, temperature=temperature)
            # t1 = time.time()
            # time_mask_gen.append((t1-t0) * 1000)
            # mask = torch.zeros((1, self.patch_groups, self.mask_size, self.mask_size))
            if self.mask_size==14:
                mask = mask14
            elif self.mask_size==7:
                mask = mask7
            else:
                mask = mask2
            _masks.append(mask)
            x_refine = self.refine_module[i](x_refine, mask, inference=inference)
            refine_ls.append(x_refine)
        # t0 = time.time()
        if self.alpha != 1:
            x_base = self.base_transform(x_base)
        if self.beta != 1:
            x_refine = self.refine_transform(x_refine)
        _, _, h, w = x_refine.shape
        x_base = F.interpolate(x_base, size=(h, w))
        out = self.relu(x_base + x_refine)
        # t1 = time.time()
        # time_transform_fusion.append((t1-t0) * 1000)
        out = self.fusion[0](out)
        return refine_ls, out, _masks

    def forward_calc_flops(self, x, temperature=1e-8, inference=False):
        b, c, h, w = x.size()
        flops = 0
        _masks = []
        x_refine = x
        refine_ls = []
        
        for i in range(len(self.base_module)):
            x_base, _flops = self.base_module[i].forward_calc_flops(x_base) if i != 0 else self.base_module[
                i].forward_calc_flops(x)
            flops += _flops
            mask, _flops = self.mask_gen[i].forward_calc_flops(x_base, temperature=temperature)
            if self.mask_size==14:
                mask = mask14
            elif self.mask_size==7:
                mask = mask7
            else:
                mask = mask2
            _masks.append(mask)
            flops += _flops
            x_refine, _flops = self.refine_module[i].forward_calc_flops(x_refine, mask, inference=inference)
            refine_ls.append(x_refine)
            flops += _flops

        c = x_base.shape[1]
        _, _, h, w = x_refine.shape
        if self.alpha != 1:
            x_base = self.base_transform(x_base)
            flops += c * x_base.shape[1] * x_base.shape[2] * x_base.shape[3]
        if self.beta != 1:
            c_in = x_refine.shape[1]
            x_refine = self.refine_transform(x_refine)
            flops += c_in * x_refine.shape[1] * x_refine.shape[2] * x_refine.shape[3]
        x_base = F.interpolate(x_base, size=(h, w))
        out = self.relu(x_base + x_refine)
        out, _flops = self.fusion[0].forward_calc_flops(out)
        flops += _flops
        return refine_ls, out, _masks, flops


class sarResNet(nn.Module):
    def __init__(self, block_base, block_refine, layers, num_classes=1000, patch_groups=1, mask_size=7, width=1.0,
                 alpha=1, beta=1, base_scale=2):
        num_channels = [int(64 * width), int(128 * width), int(256 * width), 512]
        # print(num_channels)
        self.inplanes = 64
        super(sarResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, num_channels[0], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels[0])
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = sarModule(block_base, block_refine, num_channels[0], num_channels[0] * block_base.expansion,
                                layers[0], stride=2, groups=patch_groups, mask_size=mask_size, alpha=alpha,
                                base_scale=base_scale)
        self.layer2 = sarModule(block_base, block_refine, num_channels[0] * block_base.expansion,
                                num_channels[1] * block_base.expansion, layers[1], stride=2, groups=patch_groups,
                                mask_size=mask_size, alpha=alpha, beta=beta, base_scale=base_scale)

        self.layer3 = sarModule(block_base, block_refine, num_channels[1] * block_base.expansion,
                                num_channels[2] * block_base.expansion, layers[2], stride=1, groups=patch_groups,
                                mask_size=2, alpha=alpha, beta=beta, base_scale=2)
        self.layer4 = self._make_layer(
            block_base, num_channels[2] * block_base.expansion, num_channels[3] * block_base.expansion, layers[3],
            stride=2)
        self.gappool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_channels[3] * block_base.expansion, num_classes)

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
        # t0 = time.time()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print('before layer 1:', x.shape)
        # t1 = time.time()
        # time_stem.append((t1 - t0)* 1000)
        _masks = []

        ls, x, mask = self.layer1(x, temperature=temperature, inference=inference)
        _masks.extend(mask)

        _, x, mask = self.layer2(x, temperature=temperature, inference=inference)
        _masks.extend(mask)
        # print('before layer 3:', x.shape)
        _, x, mask = self.layer3(x, temperature=temperature, inference=inference)
        _masks.extend(mask)
        # print('before layer 4:', x.shape)
        for i in range(len(self.layer4)):
            x = self.layer4[i](x)
        # t2 = time.time()
        # time_layers.append((t2 - t1) * 1000)
        x = self.gappool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # t3 = time.time()
        # time_tail.append((t3-t2) * 1000)
        return ls, x, _masks

    def forward_calc_flops(self, x, temperature=1.0, inference=False):
        flops = 0
        c_in = x.shape[1]
        x = self.conv1(x)
        flops += c_in * x.shape[1] * x.shape[2] * x.shape[3] * self.conv1.weight.shape[2] * self.conv1.weight.shape[3]
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)
        flops += x.numel() / x.shape[0] * 9
        _masks = []
        # print(x.shape)
        ls, x, mask, _flops = self.layer1.forward_calc_flops(x, temperature=temperature, inference=inference)
        _masks.extend(mask)
        flops += _flops
        # print(x.shape)
        _, x, mask, _flops = self.layer2.forward_calc_flops(x, temperature=temperature, inference=inference)
        _masks.extend(mask)
        flops += _flops
        # print(x.shape)
        _, x, mask, _flops = self.layer3.forward_calc_flops(x, temperature=temperature, inference=inference)
        flops += _flops
        _masks.extend(mask)
        # print(x.shape)
        for i in range(len(self.layer4)):
            x, _flops = self.layer4[i].forward_calc_flops(x)
            flops += _flops

        flops += x.shape[1] * x.shape[2] * x.shape[3]
        x = self.gappool(x)
        x = x.view(x.size(0), -1)
        c_in = x.shape[1]
        x = self.fc(x)
        flops += c_in * x.shape[1]

        return ls, x, _masks, flops


t21=[]
t43=[]
t54=[]
t65=[]
t76=[]

def _rearrange_features(feat, mask):
    t1 = time.time()
    b, c_f, h_f, w_f = feat[0].size()
    # 
    _, c, h, w,_,_,_ = mask.size()

    h_interval = h_f
    w_interval = w_f
    c_interval = c_f
    # x = torch.zeros(1, c * c_f, h * h_f, w * w_f)

    q = 0
    kk = 0
    pp = 0

    a = np.flatnonzero(mask[0])
    l = a.shape[0]
    t2 = time.time()
    t21.append((t2 - t1) * 1000)
    x = torch.zeros(1, c*c_f, h*h_f, w*w_f)
    for n in range(l):
        t3 = time.time()
        k = a[n] // (h*w)
        if kk != k and n != 0:
            # if q >= mask[0, k, :, :].sum():
            q = 0
            pp += 1
        t4 = time.time()
        t43.append((t4 - t3) * 1000)
        p = a[n] % (h*w)
        i = p // w
        j = p % h
        h_1 = i * h_interval
        w_1 = j * w_interval
        t5 = time.time()
        t54.append((t5 - t4) * 1000)
        t5 = time.time()
        x[0, k * c_interval:(k + 1) * c_interval, h_1:h_1 + h_interval, w_1:w_1 + w_interval] = feat[pp][q] # + residual[:, k * c_interval:(k + 1) * c_interval, h_1:h_1 + h_interval, w_1:w_1 + w_interval]
        t6 = time.time()
        t65.append((t6 - t5) * 1000)
        q += 1
        kk = k
        t7 = time.time()
        t76.append((t7 - t6) * 1000)
    
    # b, g, mask_size, _,_,_,_ = mask.size()
    # print(feat[0].size(), mask.size())
    # assert(0==1)

    return x


def sar_resnet_imgnet_alphaBase(depth, num_classes=1000, patch_groups=1, mask_size=7, width=1.0, alpha=1, beta=1,
                                base_scale=2):
    layers = {
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [4, 8, 18, 3],
        152: [5, 12, 30, 3]
    }[depth]
    block = BasicBlock if depth == 34 else Bottleneck
    block_refine = BasicBlock_refine if depth == 34 else Bottleneck_refine
    model = sarResNet(block_base=block, block_refine=block_refine, layers=layers,
                      num_classes=num_classes, patch_groups=patch_groups, mask_size=mask_size, width=width,
                      alpha=alpha, beta=beta, base_scale=base_scale)
    return model


def calc_one_group(x,i,conv1,conv2,conv3,c_out_g1,c_out_g2,c_out_g3,relu):
    
    out = F.conv2d(x, conv1.weight[i * c_out_g1:(i + 1) * c_out_g1], padding=0)
    # print(x.size(), out.size())
    # assert(0==1)
    out = relu(out)
    # print(out.size())
    out = F.conv2d(out, conv2.weight[i * c_out_g2:(i + 1) * c_out_g2], padding=1)
    out = relu(out)
    # print(out.size())
    # assert(0==1)
    out = F.conv2d(out, conv3.weight[i * c_out_g3:(i + 1) * c_out_g3], padding=0)
    # out = relu(out)
    # print(out.size())
    return out

def sar_resnet34_alphaBase_4stage_imgnet(args):
    return sar_resnet_imgnet_alphaBase(depth=34, num_classes=args.num_classes, patch_groups=args.patch_groups,
                                       mask_size=args.mask_size, alpha=args.alpha, beta=args.beta,
                                       base_scale=args.base_scale)


def sar_resnet50_alphaBase_4stage_imgnet(args):
    return sar_resnet_imgnet_alphaBase(depth=50, num_classes=args.num_classes, patch_groups=args.patch_groups,
                                       mask_size=args.mask_size, alpha=args.alpha, beta=args.beta,
                                       base_scale=args.base_scale)


if __name__ == "__main__":
    
    sar_res = sar_resnet50_alphaBase_4stage_imgnet(args)
    sar_res = sar_res.cuda(0)
    # print(sar_res)
    sar_res.eval()
    with torch.no_grad():
        print(sar_res)
        x = torch.rand(1, 3, 224, 224).cuda(0)
        fuse_module(sar_res)
        ls1, y2, _masks2 = sar_res(x, inference=True, temperature=1e-8)
        print('speed test start')
        a=[]
        b=[]
        c=[]
        d=[]
        e=[]
        f=[]
        g=[]
        h=[]
        j=[]
        k=[]
        l=[]
        m=[]
        t_base_branch=[]
        t_mask_gen=[]
        t_trans_fuse = []
        t_stem=[]
        t_tail=[]
        t_layers=[]
        t_mask_define=[]
        t_prepare_mask=[]
        t_downsample=[]
        for i in range(10):
            t = time.time()
            _, y2, _masks = sar_res(x,inference=True,temperature=1e-8)
            print(time.time() - t)
        for i in range(100):
            t = time.time()
            _, y1, _masks, flops = sar_res.forward_calc_flops(x, inference=False, temperature=1e-8)
            # assert(0==1)
            tttt = time.time()
            a.append(np.sum(time_calc))
            time_calc = []
            b.append((tttt - t) * 1000)


            tt = time.time()
            _, y2, _masks = sar_res(x,inference=True,temperature=1e-8)
            ttt = time.time()
            t_base_branch.append(np.sum(time_base_branch))
            time_base_branch = []
            t_mask_gen.append(np.sum(time_mask_gen))
            time_mask_gen = []
            t_prepare_mask.append(np.sum(time_prepare_mask))
            time_prepare_mask = []
            t_trans_fuse.append(np.sum(time_transform_fusion))
            time_transform_fusion = []
            t_layers.append(np.sum(time_layers))
            time_layers = []
            t_mask_define.append(np.sum(time_mask_define))
            time_mask_define = []
            t_downsample.append(np.sum(time_ds))
            time_ds = []
            t_stem.append(np.sum(time_stem))
            time_stem = []
            t_tail.append(np.sum(time_tail))
            time_tail = []
            c.append(np.sum(time_extract))
            time_extract = []
            d.append(np.sum(time_conv))
            time_conv = []
            e.append(np.sum(time_rearrange))
            time_rearrange = []
            f.append(np.sum(time_forward))
            time_forward = []
            g.append((ttt-tt)*1000)
            h.append(np.sum(t21))
            t21 = []
            j.append(np.sum(t43))
            t43 = []
            k.append(np.sum(t54))
            t54 = []
            l.append(np.sum(t65))
            t65 = []
            m.append(np.sum(t76))
            t76 = []
            print(i)

        print('bottleneck_group:', np.mean(a))
        print('total_group', np.mean(b))
        print('-------------------------------')
        print('time_stem:', np.mean(t_stem))
        print('time_base_bottleneck:', np.mean(t_base_branch))
        print('time_refine_bottleneck:', np.mean(f))
        print('time_downsample_in_refine:', np.mean(t_downsample))
        print('time_mask_prepare_in_refine:', np.mean(t_prepare_mask))
        print('time_mask_gen:', np.mean(t_mask_gen))
        print('time_mask_define', np.mean(t_mask_define))
        print('time_trans_fuse:', np.mean(t_trans_fuse))
        print('time_tail:', np.mean(t_tail))
        print('time_layers', np.mean(t_layers))
        print('total_time', np.mean(g), np.std(g))
        print('-------------------------------')
        print('time_refine_bottleneck:', np.mean(f))
        print('time_downsample_in_refine:', np.mean(t_downsample))
        print('time_mask_prepare_in_refine:', np.mean(t_prepare_mask))
        print('time_extract:', np.mean(c))
        print('time_conv:', np.mean(d))
        print('time_rearrange:', np.mean(e))
        print('-------------------------------')
        print('t21_in_rearrange', np.mean(h))
        print('t43_in_rearrange', np.mean(j))
        print('t54_in_rearrange', np.mean(k))
        print('t65_in_rearrange', np.mean(l))
        print('t76_in_rearrange', np.mean(m))
        print('-------------------------------')

        ls, y1, _masks1, flops = sar_res.forward_calc_flops(x,inference=False,temperature=1e-8)
        ls1, y2, _masks2 = sar_res(x, inference=True, temperature=1e-8)

        print((y1-y2).abs().sum())
        print((ls[0]-ls1[0]).abs().sum())

        # t_sim = []
        # for i in range(100):
        #     t1 = time.time()
        #     y1, _masks, flops = sar_res.forward_calc_flops(x,inference=True,temperature=1e-8)
        #     if i >= 10:
        #         t_sim.append(time.time() - t1)
        # print('TIME sim: ', np.mean(t_sim))

        # t_real = []
        # for i in range(100):
        #     t2 = time.time()
        #     y2, _masks = sar_res(x,inference=True,temperature=1e-8)
        #     print(time.time() - t2)
        #     t_real.append(time.time() - t2)
        # print('TIME real: ', np.mean(t_real))

        # print(flops/1e9)


