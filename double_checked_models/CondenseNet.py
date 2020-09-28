import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CondenseNet', 'cdn_a', 'cdn_b', 'cdn_c', 'cdn_d']


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, groups=1, activation='ReLU'):
        super(Conv, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        if activation == 'ReLU':
            # self.add_module('relu', nn.ReLU(inplace=True))
            self.add_module('activation', nn.ReLU(inplace=True))
        else:
            # self.add_module('activation', HS())
            raise NotImplementedError
        self.add_module('conv', nn.Conv2d(in_channels, out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding, bias=False,
                                          groups=groups))


class LearnedGroupConv(nn.Module):
    global_progress = 0.0

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, condense_factor=None,
                 dropout_rate=0., activation='ReLU'):
        super(LearnedGroupConv, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        if activation == 'ReLU':
            # self.relu = nn.ReLU(inplace=True)
            self.activation = nn.ReLU(inplace=True)
        else:
            # self.activation = HS()
            raise NotImplementedError
        self.dropout_rate = dropout_rate
        if self.dropout_rate > 0:
            self.drop = nn.Dropout(dropout_rate, inplace=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups=1, bias=False)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.condense_factor = condense_factor
        if self.condense_factor is None:
            self.condense_factor = self.groups
        ### Parameters that should be carefully used
        self.register_buffer('_count', torch.zeros(1))
        self.register_buffer('_stage', torch.zeros(1))
        self.register_buffer('_mask', torch.ones(self.conv.weight.size()))
        ### Check if arguments are valid
        assert self.in_channels % self.groups == 0, "group number can not be divided by input channels"
        assert self.in_channels % self.condense_factor == 0, "condensation factor can not be divided by input channels"
        assert self.out_channels % self.groups == 0, "group number can not be divided by output channels"

    def forward(self, x):
        self._check_drop()
        x = self.norm(x)
        # x = self.relu(x)
        x = self.activation(x)
        if self.dropout_rate > 0:
            x = self.drop(x)
        ### Masked output
        weight = self.conv.weight * self.mask
        return F.conv2d(x, weight, None, self.conv.stride,
                        self.conv.padding, self.conv.dilation, 1)

    def _check_drop(self):
        progress = LearnedGroupConv.global_progress
        delta = 0
        ### Get current stage
        for i in range(self.condense_factor - 1):
            if progress * 2 < (i + 1) / (self.condense_factor - 1):
                stage = i
                break
        else:
            stage = self.condense_factor - 1
        ### Check for dropping
        if not self._reach_stage(stage):
            self.stage = stage
            delta = self.in_channels // self.condense_factor
        if delta > 0:
            self._dropping(delta)
        return

    def _dropping(self, delta):
        print('LearnedGroupConv dropping')
        weight = self.conv.weight * self.mask
        ### Sum up all kernels
        ### Assume only apply to 1x1 conv to speed up
        assert weight.size()[-1] == 1
        weight = weight.abs().squeeze()
        assert weight.size()[0] == self.out_channels
        assert weight.size()[1] == self.in_channels
        d_out = self.out_channels // self.groups
        ### Shuffle weight
        weight = weight.view(d_out, self.groups, self.in_channels)
        weight = weight.transpose(0, 1).contiguous()
        weight = weight.view(self.out_channels, self.in_channels)
        ### Sort and drop
        for i in range(self.groups):
            wi = weight[i * d_out:(i + 1) * d_out, :]
            ### Take corresponding delta index
            di = wi.sum(0).sort()[1][self.count:self.count + delta]
            for d in di.data:
                self._mask[i::self.groups, d, :, :].fill_(0)
        self.count = self.count + delta

    @property
    def count(self):
        return int(self._count[0])

    @count.setter
    def count(self, val):
        self._count.fill_(val)

    @property
    def stage(self):
        return int(self._stage[0])

    @stage.setter
    def stage(self, val):
        self._stage.fill_(val)

    @property
    def mask(self):
        return self._mask

    def _reach_stage(self, stage):
        return (self._stage >= stage).all()

    @property
    def lasso_loss(self):
        if self._reach_stage(self.groups - 1):
            return 0
        weight = self.conv.weight * self.mask
        ### Assume only apply to 1x1 conv to speed up
        assert weight.size()[-1] == 1
        weight = weight.squeeze().pow(2)
        d_out = self.out_channels // self.groups
        ### Shuffle weight
        weight = weight.view(d_out, self.groups, self.in_channels)
        weight = weight.sum(0).clamp(min=1e-6).sqrt()
        return weight.sum()


class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, group_1x1, group_3x3,
                 bottleneck, condense_factor):
        super(_DenseLayer, self).__init__()
        self.group_1x1 = group_1x1
        self.group_3x3 = group_3x3
        ### 1x1 conv i --> b*k
        self.conv_1 = LearnedGroupConv(in_channels, bottleneck * growth_rate,
                                       kernel_size=1, groups=self.group_1x1,
                                       condense_factor=condense_factor,
                                       dropout_rate=0.0)
        ### 3x3 conv b*k --> k
        self.conv_2 = Conv(bottleneck * growth_rate, growth_rate,
                           kernel_size=3, padding=1, groups=self.group_3x3)

    def forward(self, x):
        x_ = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        return torch.cat([x_, x], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, growth_rate, group_1x1,
                 group_3x3, bottleneck, condense_factor):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(in_channels + i * growth_rate, growth_rate, group_1x1,
                                group_3x3, bottleneck, condense_factor)
            self.add_module('denselayer_%d' % (i + 1), layer)


class _Transition(nn.Module):
    def __init__(self):
        super(_Transition, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        return x


class CondenseNet(nn.Module):
    def __init__(self, stages, growth, group_1x1, group_3x3, bottleneck,
                 condense_factor, dropout=0.0, num_classes=1000, dataset='imagenet'):

        super(CondenseNet, self).__init__()

        self.stages = stages
        self.growth = growth
        self.group_1x1 = group_1x1
        self.group_3x3 = group_3x3
        self.bottleneck = bottleneck
        self.condense_factor = condense_factor
        assert len(self.stages) == len(self.growth)
        self.progress = 0.0
        if dataset in ['cifar10', 'cifar100']:
            self.init_stride = 1
            self.pool_size = 8
        else:
            self.init_stride = 2
            self.pool_size = 7

        self.features = nn.Sequential()
        ### Initial nChannels should be 3
        self.num_features = 2 * self.growth[0]
        ### Dense-block 1 (224x224)
        self.features.add_module('init_conv', nn.Conv2d(3, self.num_features,
                                                        kernel_size=3,
                                                        stride=self.init_stride,
                                                        padding=1,
                                                        bias=False))
        for i in range(len(self.stages)):
            ### Dense-block i
            self.add_block(i)

        ### Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        ### Linear layer
        self.classifier = nn.Linear(self.num_features, num_classes)

        ### initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        return

    def add_block(self, i):
        ### Check if ith is the last one
        last = (i == len(self.stages) - 1)
        block = _DenseBlock(
            num_layers=self.stages[i],
            in_channels=self.num_features,
            growth_rate=self.growth[i],
            group_1x1=self.group_1x1,
            group_3x3=self.group_3x3,
            bottleneck=self.bottleneck,
            condense_factor=self.condense_factor
        )
        self.features.add_module('denseblock_%d' % (i + 1), block)
        self.num_features += self.stages[i] * self.growth[i]
        if not last:
            trans = _Transition()
            self.features.add_module('transition_%d' % (i + 1), trans)
        else:
            self.features.add_module('norm_last',
                                     nn.BatchNorm2d(self.num_features))
            self.features.add_module('relu_last',
                                     nn.ReLU(inplace=True))
            self.features.add_module('pool_last',
                                     nn.AvgPool2d(self.pool_size))

    def forward(self, x, progress=None):
        if progress:
            LearnedGroupConv.global_progress = progress
        features = self.features(x)
        out = features.view(features.size(0), -1)
        if self.dropout:
            out = self.dropout(out)
        out = self.classifier(out)
        return out


def cdn_a(num_classes=1000, dropout=0.0):
    stages = '2-4-6-8-4'
    growth = '8-8-16-32-64'
    stages = list(map(int, stages.split('-')))
    growth = list(map(int, growth.split('-')))
    condense_factor = 8
    group_1x1 = 8
    group_3x3 = 8
    bottleneck = 4
    dropout = dropout
    num_classes = num_classes
    return CondenseNet(stages, growth, group_1x1, group_3x3, bottleneck,
                       condense_factor, dropout=dropout, num_classes=num_classes)


def cdn_b(num_classes=1000, dropout=0.0):
    stages = '2-4-6-8-6'
    growth = '6-12-24-48-96'
    stages = list(map(int, stages.split('-')))
    growth = list(map(int, growth.split('-')))
    condense_factor = 6
    group_1x1 = 6
    group_3x3 = 6
    bottleneck = 4
    dropout = dropout
    num_classes = num_classes
    return CondenseNet(stages, growth, group_1x1, group_3x3, bottleneck,
                       condense_factor, dropout=dropout, num_classes=num_classes)


def cdn_c(num_classes=1000, dropout=0.0):
    stages = '4-6-8-10-8'
    growth = '8-16-32-64-128'
    stages = list(map(int, stages.split('-')))
    growth = list(map(int, growth.split('-')))
    condense_factor = 8
    group_1x1 = 8
    group_3x3 = 8
    bottleneck = 4
    dropout = dropout
    num_classes = num_classes
    return CondenseNet(stages, growth, group_1x1, group_3x3, bottleneck,
                       condense_factor, dropout=dropout, num_classes=num_classes)


def cdn_d(num_classes=1000, dropout=0.0):
    stages = '4-6-8-10-8'
    growth = '8-16-32-64-128'
    stages = list(map(int, stages.split('-')))
    growth = list(map(int, growth.split('-')))
    condense_factor = 4
    group_1x1 = 4
    group_3x3 = 4
    bottleneck = 4
    dropout = dropout
    num_classes = num_classes
    return CondenseNet(stages, growth, group_1x1, group_3x3, bottleneck,
                       condense_factor, dropout=dropout, num_classes=num_classes)