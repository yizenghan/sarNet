import torch.nn as nn
from .octconv import *



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, alpha_in=0.5, alpha_out=0.5, norm_layer=None, output=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = Conv_BN_ACT(inplanes, width, kernel_size=1, alpha_in=alpha_in, alpha_out=alpha_out, norm_layer=norm_layer)
        self.conv2 = Conv_BN_ACT(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, norm_layer=norm_layer,
                                 alpha_in=0 if output else 0.5, alpha_out=0 if output else 0.5)
        self.conv3 = Conv_BN(width, planes * self.expansion, kernel_size=1, norm_layer=norm_layer,
                             alpha_in=0 if output else 0.5, alpha_out=0 if output else 0.5)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity_h = x[0] if type(x) is tuple else x
        identity_l = x[1] if type(x) is tuple else None

        x_h, x_l = self.conv1(x)
        x_h, x_l = self.conv2((x_h, x_l))
        x_h, x_l = self.conv3((x_h, x_l))

        if self.downsample is not None:
            identity_h, identity_l = self.downsample(x)

        x_h += identity_h
        x_l = x_l + identity_l if identity_l is not None else None

        x_h = self.relu(x_h)
        x_l = self.relu(x_l) if x_l is not None else None

        return x_h, x_l

    def forward_calc_flops(self, x):
        flops = 0
        identity_h = x[0] if type(x) is tuple else x
        identity_l = x[1] if type(x) is tuple else None

        x_h, x_l, _flops = self.conv1.forward_calc_flops(x)
        flops += _flops
        x_h, x_l, _flops = self.conv2.forward_calc_flops((x_h, x_l))
        flops += _flops
        x_h, x_l, _flops = self.conv3.forward_calc_flops((x_h, x_l))
        flops += _flops

        if self.downsample is not None:
            identity_h, identity_l, _flops = self.downsample.forward_calc_flops(x)
            flops += _flops

        x_h += identity_h
        x_l = x_l + identity_l if identity_l is not None else None

        x_h = self.relu(x_h)
        x_l = self.relu(x_l) if x_l is not None else None

        return x_h, x_l, flops


class OctResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(OctResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, alpha_in=0)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer, alpha_out=0, output=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, alpha_in=0.5, alpha_out=0.5, norm_layer=None, output=False):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Conv_BN(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, alpha_in=alpha_in, alpha_out=alpha_out)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, alpha_in, alpha_out, norm_layer, output))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer,
                                alpha_in=0 if output else 0.5, alpha_out=0 if output else 0.5, output=output))

        return nn.ModuleList(layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        
        x_h, x_l = self.layer1[0](x)
        for i in range(1,len(self.layer1)):
            x_h, x_l = self.layer1[i]((x_h,x_l))

        for i in range(len(self.layer2)):
            x_h, x_l = self.layer2[i]((x_h,x_l))

        for i in range(len(self.layer3)):
            x_h, x_l = self.layer3[i]((x_h,x_l))
        
        for i in range(len(self.layer4)):
            x_h, x_l = self.layer4[i]((x_h,x_l))
        x = self.avgpool(x_h)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def forward_calc_flops(self, x):
        flops = 0
        c_in = x.shape[1]
        x = self.conv1(x)
        _,c,h,w = x.shape
        flops += c_in * c * h * w * self.conv1.weight.shape[2]* self.conv1.weight.shape[3]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        flops += x.shape[1] * x.shape[2] * x.shape[3] * 9

        # print(self.layer1[0])
        x_h, x_l, _flops = self.layer1[0].forward_calc_flops(x)
        flops += _flops
        for i in range(1,len(self.layer1)):
            x_h, x_l, _flops = self.layer1[i].forward_calc_flops((x_h,x_l))
            flops += _flops

        for i in range(len(self.layer2)):
            x_h, x_l, _flops = self.layer2[i].forward_calc_flops((x_h,x_l))
            flops += _flops

        for i in range(len(self.layer3)):
            x_h, x_l, _flops = self.layer3[i].forward_calc_flops((x_h,x_l))
            flops += _flops
        
        for i in range(len(self.layer4)):
            x_h, x_l, _flops = self.layer4[i].forward_calc_flops((x_h,x_l))
            flops += _flops
        
        flops += x_h.numel() / x_h.shape[0]
        x = self.avgpool(x_h)
        
        x = x.view(x.size(0), -1)
        c_in = x.shape[1]
        x = self.fc(x)
        flops += c_in * x.shape[1]

        return x, flops


def oct_resnet26(pretrained=False, **kwargs):
    """Constructs a Octave ResNet-26 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = OctResNet(Bottleneck, [2, 2, 2, 2], **kwargs)
    return model


def oct_resnet50(pretrained=False, **kwargs):
    """Constructs a Octave ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = OctResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def oct_resnet101(pretrained=False, **kwargs):
    """Constructs a Octave ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = OctResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def oct_resnet152(pretrained=False, **kwargs):
    """Constructs a Octave ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = OctResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def oct_resnet200(pretrained=False, **kwargs):
    """Constructs a Octave ResNet-200 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = OctResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model


class OctResNet_cifar(nn.Module):

    def __init__(self, block, layers, num_classes=100, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(OctResNet_cifar, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, alpha_in=0)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer, alpha_out=0, output=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, alpha_in=0.5, alpha_out=0.5, norm_layer=None, output=False):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Conv_BN(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, alpha_in=alpha_in, alpha_out=alpha_out)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, alpha_in, alpha_out, norm_layer, output))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer,
                                alpha_in=0 if output else 0.5, alpha_out=0 if output else 0.5, output=output))

        return nn.ModuleList(layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x_h, x_l = self.layer1[0](x)
        for i in range(1,len(self.layer1)):
            x_h, x_l = self.layer1[i]((x_h,x_l))

        for i in range(len(self.layer2)):
            x_h, x_l = self.layer2[i]((x_h,x_l))

        for i in range(len(self.layer3)):
            x_h, x_l = self.layer3[i]((x_h,x_l))
        
        for i in range(len(self.layer4)):
            x_h, x_l = self.layer4[i]((x_h,x_l))
        x = self.avgpool(x_h)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def forward_calc_flops(self, x):
        c_in = x.shape[1]
        flops = 0
        x = self.conv1(x)
        _,c,h,w = x.shape
        flops += c_in * c * h * w * self.conv1.weight.shape[2]* self.conv1.weight.shape[3]
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x_h, x_l, _flops = self.layer1[0].forward_calc_flops(x)
        flops += _flops
        for i in range(1,len(self.layer1)):
            x_h, x_l, _flops = self.layer1[i].forward_calc_flops((x_h,x_l))
            flops += _flops

        for i in range(len(self.layer2)):
            x_h, x_l, _flops = self.layer2[i].forward_calc_flops((x_h,x_l))
            flops += _flops

        for i in range(len(self.layer3)):
            x_h, x_l, _flops = self.layer3[i].forward_calc_flops((x_h,x_l))
            flops += _flops
        
        for i in range(len(self.layer4)):
            x_h, x_l, _flops = self.layer4[i].forward_calc_flops((x_h,x_l))
            flops += _flops
        
        flops += x_h.numel() / x_h.shape[0]
        x = self.avgpool(x_h)
        
        x = x.view(x.size(0), -1)
        c_in = x.shape[1]
        x = self.fc(x)
        flops += c_in * x.shape[1]

        return x, flops

def oct_resnet26_cifar(args):
    """Constructs a Octave ResNet-26 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = OctResNet_cifar(Bottleneck, [2, 2, 2, 2], num_classes = args.num_classes)
    return model

def oct_resnet50_cifar(args):
    """Constructs a Octave ResNet-200 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = OctResNet_cifar(Bottleneck, [3, 4, 6, 3], num_classes = args.num_classes)
    return model

if __name__ == '__main__':
    from op_counter import measure_model
    import argparse
    import numpy as np
    import time
    parser = argparse.ArgumentParser(description='PyTorch resnet Training')
    args = parser.parse_args()
    args.num_classes = 1000
    net = oct_resnet50()#.cuda(1)
    net.eval()
    x = torch.rand(1,3,224,224)#.cuda(1)
    # y, _flops = net.forward_calc_flops(x)
    # print(_flops / 1e9)
    # example = torch.rand(1, 3, 224, 224)
    # traced_script_module = torch.jit.trace(model, example)
    # torchscript_model_optimized = optimize_for_mobile(traced_script_module)
    # torchscript_model_optimized.save("oct_r50.pt")

    
    

    t_sim = []
    for i in range(100):
        t1 = time.time()
        y = net(x)
        if i >= 10:
            t = time.time() - t1
            print(t)
            t_sim.append(t)
    print('TIME sim: ', np.mean(t_sim))
    # s = 0
    # for item in t_sim:
    #     s+=pow((item-np.mean(t_sim)),2)
    # sa = s / len(t_sim)
    print(np.std(t_sim)) 
    y, _flops = net.forward_calc_flops(x)
    print(_flops / 1e9)