import torch.nn as nn
# from torch.hub import load_state_dict_from_url
import torch
import torch.nn.functional as F
from .gumbel_softmax import GumbleSoftmax
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from .op_counter import measure_model

__all__ = ['sar_resnet']

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1,groups=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,groups=groups)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, last_relu=True,patch_groups=1):
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
        if self.downsample is not None:
            self.have_pool = True
            self.have_1x1conv2d = True
        
        self.stride = stride
        self.last_relu = last_relu

    def forward(self, x):
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
            _, c_in, h, w = x.shape
            residual = self.downsample(x)
            if self.have_pool:
                flops += 9 * c_in * h * w
            if self.have_1x1conv2d:
                flops += c_in * residual.shape[1] * residual.shape[2] * residual.shape[3]  
        
        out += residual
        if self.last_relu:
            out = self.relu(out)
        return out, flops

class Bottleneck_refine(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, last_relu=True,patch_groups=1):
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

        if not inference:
            b,c,h,w = x.shape
            g = mask.shape[1]
            m_h = mask.shape[2]
            if g > 1:
                mask1 = mask.unsqueeze(1).repeat(1,c//g,1,1,1).transpose(1,2).reshape(b,c,m_h,m_h)
            else:
                mask1 = mask.clone()
            mask1 = F.interpolate(mask1, size = (h,w))
            # print(mask1.shape, x.shape)
            out = x * mask1
            # print(mask1)
            out = self.conv1(out)
            out = self.bn1(out)
            out = self.relu(out)

            c_out = out.shape[1]
            # print(mask1.shape, mask.shape)
            if g > 1:
                mask2 = mask.unsqueeze(1).repeat(1,c_out//g,1,1,1).transpose(1,2).reshape(b,c_out,m_h,m_h)
            else:
                mask2 = mask.clone()
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
        else:
            if mask.sum() == 0.0:
                out = self.bn3(torch.zeros(residual.shape))
                out += residual
                
                if self.last_relu:
                    out = self.relu(out)
                return out

            b,c,h,w = x.shape
            
            g = mask.shape[1]
            m_h = mask.shape[2]
            if g > 1:
                mask1 = mask.unsqueeze(1).repeat(1,c//g,1,1,1).transpose(1,2).reshape(b,c,m_h,m_h)
            else:
                mask1 = mask.clone()
            mask1 = F.interpolate(mask1, size = (h,w))
            # print(mask1.shape, x.shape)
            out = x * mask1

            x_ = _extract_from_mask(out, mask)
            
            outs = []
            pp = 0

            for i in range(g):
                c_out_g = self.conv1.out_channels // g

                if mask[0,i,:,:].sum() == 0:
                    continue
                weight = self.conv1.weight
                # print(self.conv1.out_channels)
                # print(weight.size())
                # print(self.patch_groups, x_[pp].shape)
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
                outs.append(out)

                pp +=1
            
            outs = _rearrange_features(outs, mask)

            outs += residual
            if self.last_relu:
                outs = self.relu(outs)
            return outs

    def forward_calc_flops(self, x, mask, inference=False):
        # print('refine bottleneck, input shape: ', x.shape)
        residual = x
        flops = 0
        
        if self.downsample is not None:     # skip connection before mask
            c_in = x.shape[1]
            residual = self.downsample(x)
            flops += c_in * residual.shape[1] * residual.shape[2] + residual.shape[3]

        b,c,h,w = x.shape
        g = mask.shape[1]
        m_h = mask.shape[2]
        if g > 1:
            mask1 = mask.unsqueeze(1).repeat(1,c//g,1,1,1).transpose(1,2).reshape(b,c,m_h,m_h)
        else:
            mask1 = mask.clone()
        
        ratio = mask1.sum() / mask1.numel()
        # ratio = 0.69
        # print(ratio)
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
        if g > 1:
            mask2 = mask.unsqueeze(1).repeat(1,c_out//g,1,1,1).transpose(1,2).reshape(b,c_out,m_h,m_h)
        else:
            mask2 = mask.clone()
        mask2 = F.interpolate(mask2, size = (h,w))

        ratio = mask2.sum() / mask2.numel()
        # ratio = 0.69
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
        self.fc_gs.bias.data[:groups] = 0.1
        self.fc_gs.bias.data[groups:] = 5.0      
        self.gs = GumbleSoftmax()

    def forward(self, x, temperature=1.0):
        gates = self.conv3x3_gs(x)
        gates = self.pool(gates)
        gates = self.fc_gs(gates)
        gates = gates.view(x.shape[0],2,self.groups,self.mask_size,self.mask_size)
        # print(temperature)
        gates = self.gs(gates, temp=temperature, force_hard=True)
        gates = gates[:,1,:,:,:]
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
        gates = gates.view(x.shape[0],2,self.groups,self.mask_size,self.mask_size)
        # print(temperature)
        gates = self.gs(gates, temp=temperature, force_hard=True)
        gates = gates[:,1,:,:,:]
        return gates, flops

class sarModule(nn.Module):
    def __init__(self, block_base, block_refine, in_channels, out_channels, blocks, stride, groups=1,mask_size=7):
        super(sarModule, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.patch_groups = groups
        self.mask_size = mask_size
        mask_gen_list = []
        for _ in range(blocks - 1):
            mask_gen_list.append(maskGen(groups=groups,inplanes=out_channels,mask_size=mask_size))
        self.mask_gen = nn.ModuleList(mask_gen_list)
        self.base_module = self._make_layer(block_base, in_channels, out_channels, blocks - 1, 2, last_relu=False)
        self.refine_module = self._make_layer(block_refine, in_channels, out_channels, blocks - 1, 1, last_relu=False)
        self.fusion = self._make_layer(block_base, out_channels, out_channels, 1, stride=stride)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, last_relu=True):
        downsample = []
        if stride != 1:
            downsample.append(nn.AvgPool2d(3, stride=2, padding=1))
        if inplanes != planes:
            downsample.append(nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False))
            downsample.append(nn.BatchNorm2d(planes))
        # print(downsample)
        downsample = None if downsample == [] else nn.Sequential(*downsample)
        layers = []
        if blocks == 1:
            layers.append(block(inplanes, planes, stride=stride, downsample=downsample,patch_groups=self.patch_groups))
        else:
            layers.append(block(inplanes, planes, stride, downsample,patch_groups=self.patch_groups))
            for i in range(1, blocks):
                layers.append(block(planes, planes,
                                    last_relu=last_relu if i == blocks - 1 else True,patch_groups=self.patch_groups))

        return nn.ModuleList(layers)

    def forward(self, x, temperature=1e-8, inference=False):
        # print('base_module len: ', len(self.base_module))
        
        # b,c,h,w = x.size()
        # if h in [56,28]:
        #     mask = torch.ones(1, self.patch_groups, 7, 7)
        #     mask[0,0,0:7:2,0]=0.0            
        # else:
        #     mask = torch.zeros(1, self.patch_groups, 2, 2)
        #     mask[0,0,0,0]=1.0
        # z = torch.clone(x)
        # mask1 = torch.zeros(b,len(self.base_module),self.patch_groups,self.mask_size,self.mask_size)

        # for i in range(len(self.base_module)):
        #     x_base = self.base_module[i](x_base) if i!=0 else self.base_module[i](x)
        #     mask = self.mask_gen[i](x_base, temperature=temperature)
        #     mask1[:,i,:,:,:] = mask
        
        # mask2 = torch.zeros(b,len(self.base_module),self.patch_groups,self.mask_size,self.mask_size)
        _masks = []
        for i in range(len(self.base_module)):
            
            x_base = self.base_module[i](x_base) if i!=0 else self.base_module[i](x)
            mask = self.mask_gen[i](x_base, temperature=temperature)
            _masks.append(mask)
            # mask = mask1[:,i,:,:,:]
            x_refine = self.refine_module[i](x_refine, mask, inference=False) if i!=0 else self.refine_module[i](x, mask, inference=False)
            # x_refine_2 = self.refine_module[i](x_refine_2, mask, inference=True) if i!=0 else self.refine_module[i](x, mask, inference=True)
        
        # print((x_refine-x_refine_2).abs().sum())
        # assert(0==1)
        _,_,h,w = x_refine.shape
        x_base = F.interpolate(x_base, size = (h,w), mode = 'bilinear')
        out = self.relu(x_base + x_refine)
        # print(out.shape)
        out = self.fusion[0](out)
        return out, _masks

    def forward_calc_flops(self, x, temperature=1e-8, inference=False):
        b,c,h,w = x.size()
        flops = 0
        _masks = []
        for i in range(len(self.base_module)):
            x_base, _flops = self.base_module[i].forward_calc_flops(x_base) if i!=0 else self.base_module[i].forward_calc_flops(x)
            flops += _flops
            mask, _flops = self.mask_gen[i].forward_calc_flops(x_base, temperature=temperature)
            _masks.append(mask)
            flops += _flops
            x_refine, _flops = self.refine_module[i].forward_calc_flops(x_refine, mask, inference=False) if i!=0 else self.refine_module[i].forward_calc_flops(x, mask, inference=False)
            flops += _flops

        _,_,h,w = x_refine.shape
        x_base = F.interpolate(x_base, size = (h,w), mode = 'bilinear')
        out = self.relu(x_base + x_refine)
        out, _flops = self.fusion[0].forward_calc_flops(out)
        flops += _flops
        return out, _masks, flops

class sarResNet(nn.Module):
    def __init__(self, block_base, block_refine, layers, num_classes=1000, patch_groups=1, mask_size=7, width=1.0):
        num_channels = [int(64*width), int(128*width), int(256*width), 512]
        # print(num_channels)
        self.inplanes = 64
        super(sarResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, num_channels[0], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels[0])
        self.relu = nn.ReLU(inplace=True)

        self.b_conv0 = nn.Conv2d(num_channels[0], num_channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_b0 = nn.BatchNorm2d(num_channels[0])

        alpha = 2
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

        self.layer1 = sarModule(block_base, block_refine, num_channels[0], num_channels[0]*block_base.expansion, 
                               layers[0], stride=2, groups=patch_groups,mask_size=mask_size)
        self.layer2 = sarModule(block_base, block_refine, num_channels[0]*block_base.expansion,
                               num_channels[1]*block_base.expansion, layers[1], stride=2, groups=patch_groups,mask_size=mask_size)
        
        self.layer3 = sarModule(block_base, block_refine, num_channels[1]*block_base.expansion,
                               num_channels[2]*block_base.expansion, layers[2], stride=1, groups=patch_groups,mask_size=2)
        self.layer4 = self._make_layer(
            block_base, num_channels[2]*block_base.expansion, num_channels[3]*block_base.expansion, layers[3], stride=2)
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
        x, mask = self.layer1(x, temperature=temperature, inference=inference)
        _masks.extend(mask)
        # x2 = self.layer1(x, temperature=temperature, inference=True)
        # print((x1-x2).abs().sum())
        # assert(0==1)
        # print('before layer 2:', x.shape)
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

def sar_resnet(depth, num_classes=1000, patch_groups=1, mask_size=7, width=1.0):
    layers = {
        50: [3, 4, 6, 3],
        101: [4, 8, 18, 3],
        152: [5, 12, 30, 3]
    }[depth]
    model = sarResNet(block_base=Bottleneck, block_refine=Bottleneck_refine, layers=layers, 
                    num_classes=num_classes, patch_groups=patch_groups, mask_size=mask_size, width=width)
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
    
    with torch.no_grad():
        sar_res = sar_resnet(depth=50, patch_groups=4, width=1)
        # print(model)
        sar_res.eval()
        x = torch.rand(1,3,224,224)
        # y, _masks = sar_res(x,inference=False,temperature=1e-8)
        # print(len(_masks))
        # print(_masks[0].shape)

        y1, _masks, flops = sar_res.forward_calc_flops(x,inference=False,temperature=1e-8)
        print(len(_masks))
        print(_masks[9].shape)
        print(flops / 1e9)
        # y1 = sar_res(x,inference=True)
        # print((y-y1).abs().sum())

    # group = 1
    # x = torch.rand(1,256,56,56)
    
    # mask = torch.zeros(1,group, 7, 7)
    # # mask[0,0,0,0]=1
    # # mask[0,0,1,1]=1
    # # mask[0,1,1,0]=1
    

    # layer = Bottleneck_refine(256, 256, stride=1, downsample=None, last_relu=True, patch_groups=group)
    # layer.eval()
    # with torch.no_grad():
    #     y = layer(x, mask,inference=False)
    #     y2 = layer(x, mask,inference=True)
        
    #     print((y-y2).abs().sum())

# if __name__ == "__main__":
#     # sar_module = sarModule(block_base = Bottleneck, block_refine=Bottleneck_refine,
#     #                         in_channels=64,out_channels=128,blocks=6,stride=2,groups=2)
#     # # print(sar_module)
#     # x = torch.rand(1,64,56,56)
#     # y = sar_module(x)
#     # print(y.shape)
#     from op_counter import measure_model
#     sar_res = sar_resnet(depth=101)
#     # print(sar_res50)
#     x = torch.rand(1,3,224,224)
#     y = sar_res(x)
#     print(y.shape)

    # cls_ops, cls_params = measure_model(sar_res50, 224,224)
    # print(cls_params[-1]/1e6, cls_ops[-1]/1e9)
