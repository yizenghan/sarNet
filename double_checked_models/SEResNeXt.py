import torch.nn as nn
from torch.hub import load_state_dict_from_url
from .ResNeXt import ResNet

__all__ = ['se_resnext50_32x4d', 'se_resnext101_32x4d', 'se_resnext152_32x4d']


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * self.expansion, reduction)
        self.downsample = downsample
        self.stride = stride

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
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def _resnet(num_classes, arch, block, layers, pretrained, progress, dropout=0.0, **kwargs):
    model = ResNet(num_classes, block, layers, dropout=dropout, **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model


def se_resnext50_32x4d(num_classes, pretrained=False, progress=True, dropout=0.0, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet(num_classes, 'resnext50_32x4d', SEBottleneck, [3, 4, 6, 3], pretrained, progress,
                   dropout=dropout, **kwargs)


def se_resnext101_32x4d(num_classes, pretrained=False, progress=True, dropout=0.0, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet(num_classes, 'resnext101_32x4d', SEBottleneck, [3, 4, 23, 3], pretrained, progress,
                   dropout=dropout, **kwargs)


def se_resnext152_32x4d(num_classes, pretrained=False, progress=True, dropout=0.0, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet(num_classes, 'resnext101_32x4d', SEBottleneck, [3, 8, 36, 3], pretrained, progress,
                   dropout=dropout, **kwargs)
