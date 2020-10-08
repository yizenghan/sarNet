import torch.nn as nn
# from torch.hub import load_state_dict_from_url
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

__all__ = ['sar_resnet_nomask']

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
            if len(self.downsample) > 1:
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
        flops += c_in * c_out * h * w  

        out = self.bn1(out)
        out = self.relu(out)

        c_in = c_out
        out = self.conv2(out)
        _,c_out,h,w = out.shape
        flops += c_in * c_out * h * w * 9 

        out = self.bn2(out)
        out = self.relu(out)

        c_in = c_out
        out = self.conv3(out)
        _,c_out,h,w = out.shape
        flops += c_in * c_out * h * w 
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
        # print(self.downsample)
        self.stride = stride
        self.last_relu = last_relu
        self.patch_groups = patch_groups

    def forward(self, x):
        residual = x
        if self.downsample is not None:     # skip connection before mask
            residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        if self.last_relu:
            out = self.relu(out)

        return out
        

    def forward_calc_flops(self, x):
        # print('refine bottleneck, input shape: ', x.shape)
        residual = x
        flops = 0
        
        if self.downsample is not None:     # skip connection before mask
            c_in = x.shape[1]
            residual = self.downsample(x)
            flops += c_in * residual.shape[1] * residual.shape[2] * residual.shape[3]

        c_in = x.shape[1]
        out = self.conv1(x)
        flops +=  c_in * out.shape[1] * out.shape[2] * out.shape[3] / self.conv1.groups
        out = self.bn1(out)
        out = self.relu(out)
        
        c_in = out.shape[1]
        out = self.conv2(out)
        flops += c_in * out.shape[1] * out.shape[2] * out.shape[3] * 9 / self.conv2.groups
        out = self.bn2(out)
        out = self.relu(out)

        c_in = out.shape[1]
        out = self.conv3(out)
        flops += c_in * out.shape[1] * out.shape[2] * out.shape[3] / self.conv3.groups
        out = self.bn3(out)
        out += residual
        if self.last_relu:
            out = self.relu(out)
        return out, flops
 
class sarModule(nn.Module):
    def __init__(self, block_base, block_refine, in_channels, out_channels, blocks, stride, groups=1, alpha=1, beta=1):
        super(sarModule, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.patch_groups = groups

        self.base_module = self._make_layer(block_base, in_channels, out_channels, blocks - 1, 2, last_relu=False)
        self.refine_module = self._make_layer(block_refine, in_channels, out_channels // alpha, max(1, blocks // beta - 1), 1, last_relu=True)
        self.little_e = nn.Sequential(
            nn.Conv2d(out_channels // alpha, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
            )
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

    def forward(self, x):
        for i in range(len(self.base_module)):
            x_base = self.base_module[i](x_base) if i!=0 else self.base_module[i](x)
        for i in range(len(self.refine_module)):
            x_refine = self.refine_module[i](x_refine) if i!=0 else self.refine_module[i](x)

        x_refine = self.little_e(x_refine)
        _,_,h,w = x_refine.shape
        x_base = F.interpolate(x_base, scale_factor=2, mode = 'bilinear', align_corners=False)
        out = self.relu(x_base + x_refine)
        out = self.fusion[0](out)
        return out

    def forward_calc_flops(self, x):
        b,c,h,w = x.size()
        flops = 0
        for i in range(len(self.base_module)):
            x_base, _flops = self.base_module[i].forward_calc_flops(x_base) if i!=0 else self.base_module[i].forward_calc_flops(x)
            flops += _flops
        for i in range(len(self.refine_module)):
            x_refine, _flops = self.refine_module[i].forward_calc_flops(x_refine) if i!=0 else self.refine_module[i].forward_calc_flops(x)
            flops += _flops

        c_in = x_refine.shape[1]
        x_refine = self.little_e(x_refine)
        flops += c_in * x_refine.shape[1] * x_refine.shape[2] * x_refine.shape[3]
        
        # _,_,h,w = x_refine.shape
        x_base = F.interpolate(x_base, scale_factor=2, mode = 'bilinear', align_corners=False)
        out = self.relu(x_base + x_refine)
        out, _flops = self.fusion[0].forward_calc_flops(out)
        flops += _flops
        return out, flops

class sarResNet(nn.Module):
    def __init__(self, block_base, block_refine, layers, num_classes=1000, patch_groups=1, width=1.0, alpha=1, beta=1):
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

        # alpha = 2
        self.l_conv0 = nn.Conv2d(num_channels[0], num_channels[0] // alpha,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_l0 = nn.BatchNorm2d(num_channels[0] // alpha)
        self.l_conv1 = nn.Conv2d(num_channels[0] // alpha, num_channels[0] // alpha, 
                                 kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_l1 = nn.BatchNorm2d(num_channels[0] // alpha)
        self.l_conv2 = nn.Conv2d(num_channels[0] // alpha, num_channels[0], kernel_size=1, stride=1, bias=False)
        self.bn_l2 = nn.BatchNorm2d(num_channels[0])

        self.bl_init = nn.Conv2d(num_channels[0], num_channels[0], kernel_size=1, stride=1, bias=False)
        self.bn_bl_init = nn.BatchNorm2d(num_channels[0])

        self.layer1 = sarModule(block_base, block_refine, num_channels[0], num_channels[0]*block_base.expansion, 
                               layers[0], stride=2, groups=patch_groups, alpha=alpha, beta=beta)
        self.layer2 = sarModule(block_base, block_refine, num_channels[0]*block_base.expansion,
                               num_channels[1]*block_base.expansion, layers[1], stride=2, groups=patch_groups, 
                               alpha=alpha, beta=beta)
        
        self.layer3 = sarModule(block_base, block_refine, num_channels[1]*block_base.expansion,
                               num_channels[2]*block_base.expansion, layers[2], stride=1, groups=patch_groups, 
                               alpha=alpha, beta=beta)
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

    def forward(self, x):
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
        # print(x.shape)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        for i in range(len(self.layer4)):
            x = self.layer4[i](x)

        x = self.gappool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def forward_calc_flops(self, x):
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

        
        x, _flops = self.layer1.forward_calc_flops(x)
        flops += _flops
        
        x, _flops = self.layer2.forward_calc_flops(x)
        flops += _flops

        x, _flops = self.layer3.forward_calc_flops(x)
        flops += _flops

        for i in range(len(self.layer4)):
            x, _flops = self.layer4[i].forward_calc_flops(x)
            flops += _flops

        flops += x.shape[1] * x.shape[2] * x.shape[3]
        x = self.gappool(x)
        x = x.view(x.size(0), -1)
        c_in = x.shape[1]
        x = self.fc(x)
        flops += c_in * x.shape[1]

        return x, flops

def sar_resnet_nomask(depth, num_classes=1000, patch_groups=1, width=1.0, alpha=1, beta=1):
    layers = {
        50: [3, 4, 6, 3],
        101: [4, 8, 18, 3],
        152: [5, 12, 30, 3]
    }[depth]
    model = sarResNet(block_base=Bottleneck, block_refine=Bottleneck_refine, layers=layers, 
                    num_classes=num_classes, patch_groups=patch_groups, width=width, alpha=alpha, beta=beta)
    return model



if __name__ == "__main__":
    
    from op_counter import measure_model
    
    # print(sar_res)
    
    with torch.no_grad():
        sar_res = sar_resnet_nomask(depth=50, patch_groups=1, width=1, alpha=2, beta=1)
        cls_ops, cls_params = measure_model(sar_res, 224, 224)
        print(cls_ops[-1]/1e9, cls_params[-1]/1e6)
        # print(model)
        sar_res.eval()
        x = torch.rand(1,3,224,224)
        # y, _masks = sar_res(x,inference=False,temperature=1e-8)
        # print(len(_masks))
        # print(_masks[0].shape)

        y1, flops = sar_res.forward_calc_flops(x)
        print(flops / 1e9)
        # y1 = sar_res(x,inference=True)
        # print((y-y1).abs().sum())
