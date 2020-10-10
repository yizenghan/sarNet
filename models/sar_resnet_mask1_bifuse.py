import torch.nn as nn
# from torch.hub import load_state_dict_from_url
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

__all__ = ['sar_resnet_mask1_bifuse']

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
    def __init__(self, block_base, block_refine, in_channels, out_channels, blocks, stride, groups=1, alpha=1.0):
        super(sarModule, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.patch_groups = groups
        assert(alpha in [0.125, 0.25, 0.5, 0.75, 1.0])
        self.base_module = self._make_layer(block_base, in_channels, int(out_channels*alpha), blocks - 1, 2)
        self.refine_module = self._make_layer(block_refine, in_channels, out_channels, blocks - 1, 1)
        print(alpha)
        if alpha < 1:
            conv_b2r = []
            conv_r2b = []
            for _ in range(blocks - 1):
                conv_b2r.append(
                    nn.Sequential(
                        nn.Conv2d(int(out_channels*alpha), out_channels, kernel_size=1, padding=0, stride=1, bias=False),
                        nn.BatchNorm2d(out_channels)
                    ))
                conv_r2b.append(
                    nn.Sequential(
                        nn.Conv2d(out_channels,int(out_channels*alpha), kernel_size=1, padding=0, stride=1, bias=False),
                        nn.BatchNorm2d(int(out_channels*alpha))
                    ))
            self.conv_b2r = nn.ModuleList(conv_b2r)
            self.conv_r2b = nn.ModuleList(conv_r2b)

            self.base_transform = nn.Sequential(
                nn.Conv2d(int(out_channels*alpha), out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.alpha = alpha
        self.fusion = self._make_layer(block_base, out_channels, out_channels, 1, stride=stride)

        self.up_b2r = nn.Upsample(scale_factor=2, mode='nearest')
        self.down_r2b = nn.AvgPool2d(3, stride=2, padding=1)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = []
        if stride != 1:
            downsample.append(nn.AvgPool2d(3, stride=2, padding=1))
        if inplanes != planes:
            downsample.append(nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False))
            downsample.append(nn.BatchNorm2d(planes))
        # print(downsample)
        downsample = None if downsample == [] else nn.Sequential(*downsample)
        layers = []
        if blocks == 1:     # fusion block
            layers.append(block(inplanes, planes, stride=stride, downsample=downsample,
                                patch_groups=self.patch_groups, last_relu=True))
        else:
            layers.append(block(inplanes, planes, stride, downsample,
                                patch_groups=self.patch_groups,
                                last_relu=False))
            for _ in range(1, blocks):
                layers.append(block(planes, planes,
                                    last_relu=False, 
                                    patch_groups=self.patch_groups))

        return nn.ModuleList(layers)

    def forward(self, x):
        for i in range(len(self.base_module)):
            
            x_base = self.base_module[i](x_base) if i!=0 else self.base_module[i](x)
            x_refine = self.refine_module[i](x_refine) if i!=0 else self.refine_module[i](x)
            
            if i < len(self.base_module) - 1:
                if self.alpha < 1:
                    x_b2r = self.conv_b2r[i](x_base)
                    x_r2b = self.conv_r2b[i](x_refine)
                else:
                    x_b2r = x_base
                    x_r2b = x_refine
                x_b2r = self.up_b2r(x_b2r)
                x_r2b = self.down_r2b(x_r2b)
                x_base += x_r2b
                x_refine += x_b2r
                x_base = self.relu(x_base)
                x_refine = self.relu(x_refine)
            elif self.alpha < 1:
                x_base = self.base_transform(x_base)
            
        _,_,h,w = x_refine.shape
        x_base = self.up_b2r(x_base)
        out = self.relu(x_base + x_refine)
        out = self.fusion[0](out)
        return out

    def forward_calc_flops(self, x):
        b,c,h,w = x.size()
        flops = 0
        for i in range(len(self.base_module)):
            x_base, _flops = self.base_module[i].forward_calc_flops(x_base) if i!=0 else self.base_module[i].forward_calc_flops(x)
            flops += _flops
            x_refine, _flops = self.refine_module[i].forward_calc_flops(x_refine) if i!=0 else self.refine_module[i].forward_calc_flops(x)
            flops += _flops

            if i < len(self.base_module) - 1:
                if self.alpha < 1:
                    c_in = x_base.shape[1]
                    x_b2r = self.conv_b2r[i](x_base)
                    flops += c_in * x_b2r.shape[1]* x_b2r.shape[2]* x_b2r.shape[3]

                    c_in = x_refine.shape[1]
                    x_r2b = self.conv_r2b[i](x_refine)
                    flops += c_in * x_r2b.shape[1]* x_r2b.shape[2]* x_r2b.shape[3]
                else:
                    x_b2r = x_base
                    x_r2b = x_refine
                x_b2r = self.up_b2r(x_b2r)
                x_r2b = self.down_r2b(x_r2b)
                flops += x_r2b.numel() / b * 9
                x_base += x_r2b
                x_refine += x_b2r
                x_base = self.relu(x_base)
                x_refine = self.relu(x_refine)
            elif self.alpha < 1:
                c_in = x_base.shape[1]
                x_base = self.base_transform(x_base)
                flops += c_in * x_base.shape[1] * x_base.shape[2] * x_base.shape[3]
            
        _,c,h,w = x_refine.shape
        x_base = self.up_b2r(x_base)
        out = self.relu(x_base + x_refine)
        out, _flops = self.fusion[0].forward_calc_flops(out)
        flops += _flops
        return out, flops

class sarResNet(nn.Module):
    def __init__(self, block_base, block_refine, layers, num_classes=1000, patch_groups=1, width=1.0, alpha=1.0):
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
        self.l_conv0 = nn.Conv2d(num_channels[0], num_channels[0] //1,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_l0 = nn.BatchNorm2d(num_channels[0] // 1)
        self.l_conv1 = nn.Conv2d(num_channels[0] // 1, num_channels[0] //
                                 1, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_l1 = nn.BatchNorm2d(num_channels[0] // 1)
        self.l_conv2 = nn.Conv2d(num_channels[0] // 1, num_channels[0], kernel_size=1, stride=1, bias=False)
        self.bn_l2 = nn.BatchNorm2d(num_channels[0])

        self.bl_init = nn.Conv2d(num_channels[0], num_channels[0], kernel_size=1, stride=1, bias=False)
        self.bn_bl_init = nn.BatchNorm2d(num_channels[0])

        self.layer1 = sarModule(block_base, block_refine, num_channels[0], num_channels[0]*block_base.expansion, 
                               layers[0], stride=2, groups=patch_groups, alpha=alpha)
        self.layer2 = sarModule(block_base, block_refine, num_channels[0]*block_base.expansion,
                               num_channels[1]*block_base.expansion, layers[1], stride=2, groups=patch_groups, alpha=alpha)
        
        self.layer3 = sarModule(block_base, block_refine, num_channels[1]*block_base.expansion,
                               num_channels[2]*block_base.expansion, layers[2], stride=1, groups=patch_groups, alpha=alpha)
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

        x = self.layer1(x)
        
        x = self.layer2(x)
        
        # print('before layer 3:', x.shape)
        x = self.layer3(x)
        
        # print('before layer 4:', x.shape)
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

def sar_resnet_mask1_bifuse(depth, num_classes=1000, patch_groups=1, width=1.0, alpha=1):
    layers = {
        50: [3, 4, 6, 3],
        101: [4, 8, 18, 3],
        152: [5, 12, 30, 3]
    }[depth]
    model = sarResNet(block_base=Bottleneck, block_refine=Bottleneck_refine, layers=layers, 
                    num_classes=num_classes, patch_groups=patch_groups, width=width, alpha=alpha)
    return model



if __name__ == "__main__":
    
    from op_counter import measure_model
    
    # print(sar_res)
    
    with torch.no_grad():
        sar_res = sar_resnet_mask1_bifuse(depth=50, patch_groups=1, width=1, alpha=1)
        print(sar_res)
        # y = sar_res()
        sar_res.eval()
        x = torch.rand(1,3,224,224)
        y = sar_res(x)
        # print(len(_masks))
        # print(_masks[0].shape)

        y1, flops = sar_res.forward_calc_flops(x)
        print(flops / 1e9)
        # y1 = sar_res(x,inference=True)
        # print((y-y1).abs().sum())

