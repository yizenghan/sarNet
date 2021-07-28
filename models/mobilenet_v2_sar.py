from torch import nn
# from .utils import load_state_dict_from_url
import torch
import numpy as np
import matplotlib.pyplot as plt 
import torch.nn.functional as F
from .gumbel_softmax import GumbleSoftmax

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}

class AttentionLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(AttentionLayer, self).__init__()
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
        return attention.view(b,c,1,1)
    
    def forward_calc_flops(self, x):
        b, c, h, w = x.size()
        flops = c*h*w
        # print('jjj')
        y = self.avg_pool(x).view(b,c)
        attention = self.fc(y)
        flops += c*c//self.reduction*2 + c
        return attention.view(b,c,1,1), flops


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        super(ConvBNReLU, self).__init__()
        
        padding = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )
        self.flops_per_pixel = in_planes * out_planes * kernel_size**2 / groups
    
    def forward(self, x):
        return self.conv(x)
    
    def forward_calc_flops(self,x):
        x = self.conv(x)
        flops = self.flops_per_pixel * x.shape[2] * x.shape[3]
        
        return x, flops
        


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup


        # layers = []
        self.expand_ratio = expand_ratio
        if expand_ratio != 1:
            # pw
            # layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
            self.conv_expand = ConvBNReLU(inp, hidden_dim, kernel_size=1)
        
        self.dwc = ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim)
        self.pwc = nn.Sequential(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
            )
        
        # layers.extend([
        #     # dw
        #     ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
        #     # pw-linear
        #     nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
        #     nn.BatchNorm2d(oup),
        # ])
        # self.conv = nn.ModuleList(layers)

    def forward(self, x):
        residual = x
        if self.expand_ratio != 1:
            x = self.conv_expand(x)
        x = self.dwc(x)
        x = self.pwc(x)
        
        if self.use_res_connect:
            return x + residual
        else:
            return x
        
    def forward_calc_flops(self, x):
        flops = 0
        residual = x
        if self.expand_ratio != 1:
            c_in = x.shape[1]
            x = self.conv_expand(x)
            _,c_out,h,w = x.shape
            flops += c_in * c_out * h * w
        
        x = self.dwc(x)
        flops += x.shape[1] * x.shape[2] * x.shape[3] * 9
        
        c_in = x.shape[1]
        x = self.pwc(x)
        _,c_out,h,w = x.shape
        flops += c_in * c_out * h * w
        
        if self.use_res_connect:
            return x + residual, flops
        else:
            return x, flops

class InvertedResidual_base(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, is_first=False):
        super(InvertedResidual_base, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1) if is_first else None
        # layers = []
        self.expand_ratio = expand_ratio
        if expand_ratio != 1:
            # pw
            # layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
            self.conv_expand = ConvBNReLU(inp, hidden_dim, kernel_size=1)
        
        self.dwc = ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim)
        self.pwc = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
            )
    

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            x = self.downsample(x)
        
        if self.expand_ratio != 1:
            x = self.conv_expand(x)
        x = self.dwc(x)
        x = self.pwc(x)
        
        if self.use_res_connect:
            return x + residual
        else:
            return x
        
    def forward_calc_flops(self, x):
        flops = 0
        residual = x
        if self.downsample is not None:
            x = self.downsample(x)
            flops += x.shape[1] * x.shape[2] * x.shape[3] * 9
        
        if self.expand_ratio != 1:
            c_in = x.shape[1]
            x = self.conv_expand(x)
            _,c_out,h,w = x.shape
            flops += c_in * c_out * h * w
        
        x = self.dwc(x)
        flops += x.shape[1] * x.shape[2] * x.shape[3] * 9
        
        c_in = x.shape[1]
        x = self.pwc(x)
        _,c_out,h,w = x.shape
        flops += c_in * c_out * h * w
        
        if self.use_res_connect:
            return x + residual, flops
        else:
            return x, flops


class InvertedResidual_refine(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, patch_groups=1):
        super(InvertedResidual_refine, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        # self.downsample = nn.AvgPool2d(3, stride=2, padding=1) if is_first else None
        # layers = []
        self.expand_ratio = expand_ratio
        if expand_ratio != 1:
            # pw
            # layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
            self.conv_expand = ConvBNReLU(inp, hidden_dim, kernel_size=1)
        self.patch_groups = patch_groups
        self.dwc = ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim)
        self.pwc = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False, groups=patch_groups),
            nn.BatchNorm2d(oup)
            )
    

    def forward(self, x, mask):
        residual = x
        
        if self.expand_ratio != 1:
            x = self.conv_expand(x)
        
        # print(x.shape, mask.shape)
        b,c,h,w = x.shape
        g = mask.shape[1]
        m_h = mask.shape[2]
        mask1 = mask.clone()
        if g > 1:
            mask1 = mask1.unsqueeze(1).repeat(1,c//g,1,1,1).transpose(1,2).reshape(b,c,m_h,m_h)
            
        mask1 = F.interpolate(mask1, size = (h,w))
        # print(mask1.shape, x.shape)
        x = x * mask1
        x = self.dwc(x)
        
        _,c_in,h,w = x.shape
        mask2 = mask.clone()
        if g > 1:
            mask2 = mask2.unsqueeze(1).repeat(1,c_in//g,1,1,1).transpose(1,2).reshape(b,c_in,m_h,m_h)
        mask2 = F.interpolate(mask2, size = (h,w))
        x = x * mask2
        x = self.pwc(x)
        if self.use_res_connect:
            return x + residual
        else:
            return x

    def forward_calc_flops(self, x, mask):
        flops = 0
        residual = x
        if self.expand_ratio != 1:
            c_in = x.shape[1]
            x = self.conv_expand(x)
            _,c_out,h,w = x.shape
            flops += c_in * c_out * h * w
        ratio = mask.sum() / mask.numel()
        # ratio = 0.5
        # print(ratio)
        b,c,h,w = x.shape
        g = mask.shape[1]
        m_h = mask.shape[2]
        mask1 = mask.clone()
        if g > 1:
            mask1 = mask1.unsqueeze(1).repeat(1,c//g,1,1,1).transpose(1,2).reshape(b,c,m_h,m_h)
        mask1 = F.interpolate(mask1, size = (h,w))
        # print(mask1.shape, x.shape)
        x = x * mask1
        x = self.dwc(x)
        flops += ratio * x.shape[1] * x.shape[2] * x.shape[3] * 9

        _,c_in,h,w = x.shape
        mask2 = mask.clone()
        if g > 1:
            mask2 = mask2.unsqueeze(1).repeat(1,c_in//g,1,1,1).transpose(1,2).reshape(b,c_in,m_h,m_h)

        mask2 = F.interpolate(mask2, size = (h,w))
        x = x * mask2
        x = self.pwc(x)
        _,c_out,h,w = x.shape
        flops += ratio * c_in * c_out * h * w / self.patch_groups

        if self.use_res_connect:
            return x + residual, flops
        else:
            return x, flops

# class SARModule(nn.Module):
#     def __init__(self, t,c,n,s, input_channel, output_channel):
#         super(SARModule, self).__init__()

#         blocks = []
#         for i in range(n):
#             stride = s if i == 0 else 1
#             blocks.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
#             input_channel = output_channel

#         self.sar_module = nn.ModuleList(blocks)

#     def forward(self,x):
#         for i in range(len(self.sar_module)):
#             x = self.sar_module[i](x)
#         return x
#     def forward_calc_flops(self, x):
#         flops = 0
#         for i in range(len(self.sar_module)):
#             x, _flops = self.sar_module[i].forward_calc_flops(x)
#             flops += _flops
#         return x, flops

class maskGen(nn.Module):
    def __init__(self, groups=1, inplanes=64, mask_size=7):
        super(maskGen,self).__init__()
        self.groups = groups
        self.mask_size = mask_size
        if mask_size == 14:
            print('Mask size:', mask_size)
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

        # print(self.groups, x.shape, gates.shape)
        # assert(0==1)
        gates = gates.view(x.shape[0],self.groups,2,self.mask_size,self.mask_size)

        # for i in range(gates.shape[1]):
        #     print(gates[0,i,:,:,:])
        #     print('hhh')
        # print(temperature)
        # assert(0==1)
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



class SARModule(nn.Module):
    def __init__(self, t,c,n,s, input_channel, output_channel,
                 alpha=2, beta=1,
                 patch_groups=1, mask_size=7):
        super(SARModule, self).__init__()
        blocks_base = []
        output_channel_base = int(output_channel // alpha)
        input_channel_base = input_channel
        for i in range(n):
            stride = s if i == 0 else 1
            is_first = True if i == 0 else False
            blocks_base.append(InvertedResidual_base(input_channel_base, output_channel_base, stride, expand_ratio=t, is_first=is_first))
            input_channel_base = output_channel_base
        self.base_branch = nn.ModuleList(blocks_base)
        self.alpha = alpha
        if alpha != 1:
            self.convert_base = ConvBNReLU(output_channel_base, output_channel)
        mask_gen_list = []
        self.mask_size = mask_size
        for _ in range(n):
            mask_gen_list.append(maskGen(groups=patch_groups,inplanes=output_channel_base,mask_size=mask_size))
        self.mask_gen = nn.ModuleList(mask_gen_list)
        blocks_refine = []
        input_channel_refine = input_channel
        output_channel_refine = int(output_channel * beta)
        for i in range(n):
            stride = s if i == 0 else 1
            is_first = True if i == 0 else False
            blocks_refine.append(InvertedResidual_refine(input_channel_refine, output_channel_refine, stride, expand_ratio=t,
                                                         patch_groups=patch_groups))
            input_channel_refine = output_channel_refine
        self.refine_branch = nn.ModuleList(blocks_refine)
        self.beta = beta
        if beta != 1:
            self.convert_refine = ConvBNReLU(output_channel_refine, output_channel)
        self.att_gen = AttentionLayer(channel=output_channel, reduction=16) 
        self.fuse_1x1 = ConvBNReLU(output_channel,output_channel,kernel_size=1)
    def forward(self,x, temperature=1e-6):
        # print
        _masks = []
        x_refine = x
        # print(x.shape, self.mask_size)
        for i in range(len(self.base_branch)):
            x_base = self.base_branch[i](x) if i==0 else self.base_branch[i](x_base)
            mask = self.mask_gen[i](x_base, temperature=temperature)
            _masks.append(mask)
            x_refine = self.refine_branch[i](x_refine, mask) 
        if self.alpha != 1:
            x_base = self.convert_base(x_base)
        if self.beta != 1:
            x_refine= self.convert_refine(x_refine)
        att = self.att_gen(x_base)
        x_base = F.interpolate(x_base, scale_factor = 2)
        output = self.fuse_1x1(att*x_base + (1-att)*x_refine)
        return output, _masks

    def forward_calc_flops(self, x, temperature=1e-6):
        flops = 0
        _masks = []
        x_refine = x
        for i in range(len(self.base_branch)):
            x_base, _flops = self.base_branch[i].forward_calc_flops(x) if i==0 else self.base_branch[i].forward_calc_flops(x_base)
            flops += _flops
            mask, _flops = self.mask_gen[i].forward_calc_flops(x_base, temperature=temperature)
            _masks.append(mask)
            flops += _flops
            x_refine, _flops = self.refine_branch[i].forward_calc_flops(x_refine, mask) 
            flops += _flops
        if self.alpha != 1:
            x_base, _flops = self.convert_base.forward_calc_flops(x_base)
            flops += _flops
        if self.beta != 1:
            x_refine, _flops = self.convert_refine.forward_calc_flops(x_refine)
            flops += _flops
        att,_flops = self.att_gen.forward_calc_flops(x_base)
        flops += _flops
        x_base = F.interpolate(x_base, scale_factor = 2)
        output, _flops = self.fuse_1x1.forward_calc_flops(att*x_base + (1-att)*x_refine)
        flops += _flops
        return output, _masks, flops


class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None,
                 alpha=2,
                 beta=1,
                 patch_groups=2):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet

        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        # features = [ConvBNReLU(3, input_channel, stride=2)]
        self.conv0 = ConvBNReLU(3, input_channel, stride=2)


        # building inverted residual blocks
        t, c, _, _ = inverted_residual_setting[0]
        output_channel = _make_divisible(c * width_mult, round_nearest)
        self.first_block = block(input_channel, output_channel, stride=1, expand_ratio=t)
        input_channel = output_channel
        sar_modules = list()
        for i in range(1,5):
            t, c, n, s = inverted_residual_setting[i]
            output_channel = _make_divisible(c * width_mult, round_nearest)
            if i == 1:
                mask_size = 14
            elif i == 2:
                mask_size = 7
            else:
                mask_size = 2
            sar_module = SARModule(t,c,n,s, input_channel, output_channel, 
                                   alpha=alpha, beta=beta, 
                                   patch_groups=patch_groups,
                                   mask_size=mask_size)
            input_channel = output_channel
            sar_modules.append(sar_module)

        self.sar_modules = nn.ModuleList(sar_modules)

        block_last2 = list()
        for j in range(5,7):
            t, c, n, s = inverted_residual_setting[j]
            output_channel = _make_divisible(c * width_mult, round_nearest)

            for i in range(n):
                stride = s if i == 0 else 1
                block_last2.append(block(input_channel, output_channel, stride=stride, expand_ratio=t))
                input_channel = output_channel

        self.block_last2 = nn.ModuleList(block_last2)
        # for t, c, n, s in inverted_residual_setting:
        #     output_channel = _make_divisible(c * width_mult, round_nearest)
        #     # 
        #     for i in range(n):
        #         stride = s if i == 0 else 1
        #         features.append(block(input_channel, output_channel, stride, expand_ratio=t))
        #         input_channel = output_channel
        # building last several layers
        self.conv_last = ConvBNReLU(input_channel, self.last_channel, kernel_size=1)
        # features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        # self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for k,m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None and 'gs' not in k:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # def _forward_impl(self, x):
    #     # This exists since TorchScript doesn't support inheritance, so the superclass method
    #     # (this one) needs to have a name other than `forward` that can be accessed in a subclass
    #     x = self.features(x)
    #     x = x.mean([2, 3])
    #     x = self.classifier(x)
    #     return x

    def forward(self, x, temperature=1.0, inference=False):
        # return self._forward_impl(x)
        x = self.conv0(x)
        x = self.first_block(x)
        _masks = []
        for i in range(len(self.sar_modules)):
            # print('start a sar module', x.shape)
            x, masks = self.sar_modules[i](x, temperature=temperature)
            _masks.extend(masks)

        for i in range(len(self.block_last2)):
            x = self.block_last2[i](x)

        x = self.conv_last(x)
        x = x.mean([2, 3])
        x = self.classifier(x)

        return x, _masks

    def forward_calc_flops(self, x, temperature=1.0, inference=False):
        # return self._forward_impl(x)
        # c = x.shape[1]
        x, flops = self.conv0.forward_calc_flops(x)

        x, _flops = self.first_block.forward_calc_flops(x)
        flops += _flops
        _masks = []
        for i in range(len(self.sar_modules)):
            x, masks, _flops = self.sar_modules[i].forward_calc_flops(x, temperature=temperature)
            flops += _flops
            _masks.extend(masks)

        for i in range(len(self.block_last2)):
            x, _flops= self.block_last2[i].forward_calc_flops(x)
            flops += _flops

        x, _flops = self.conv_last.forward_calc_flops(x)
        flops += _flops

        x = x.mean([2, 3])

        c_in = x.shape[1]
        x = self.classifier(x)
        flops += c_in * x.shape[1]

        return x, _masks, flops


def mobilenet_v2_width10_sar(args, **kwargs):
    model = MobileNetV2(width_mult=1.0,
                        alpha=args.alpha,
                        beta=args.beta,
                        patch_groups=args.patch_groups, **kwargs)
    return model

def mobilenet_v2_width13_sar(args, **kwargs):
    model = MobileNetV2(width_mult=1.3,
                        alpha=args.alpha,
                        beta=args.beta,
                        patch_groups=args.patch_groups,**kwargs)
    return model

def mobilenet_v2_width14_sar(args, **kwargs):
    model = MobileNetV2(width_mult=1.4,
                        alpha=args.alpha,
                        beta=args.beta,
                        patch_groups=args.patch_groups,**kwargs)
    return model

if __name__ == "__main__":
    import argparse
    from op_counter import measure_model
    parser = argparse.ArgumentParser(description='PyTorch resnet Training')
    args = parser.parse_args()

    args.num_classes = 1000
    args.patch_groups = 2
    args.alpha = 2
    args.beta = 1
    net = mobilenet_v2_width10_sar(args)

    net.eval()
    with torch.no_grad():
        # cls_ops, cls_params = measure_model(net, 224,224)
        # print(cls_params[-1]/1e6, cls_ops[-1]/1e9)

        x = torch.rand(1,3,224,224)

        y0,_ = net(x)
        y1,_, flops = net.forward_calc_flops(x)

        print(flops/ 1e9)