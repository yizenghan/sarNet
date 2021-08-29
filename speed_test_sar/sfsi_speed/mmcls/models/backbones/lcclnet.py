import logging

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint

from ..registry import BACKBONES
from ..utils import build_norm_layer, GaussianKernel, GumbelSigmoid
from mmcls.utils import is_host, Sparsity

VIS = False

def vis_raw_im(x):
    from matplotlib import pyplot as plt
    x = x[0].cpu().numpy().transpose(1,2,0)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for i in range(3):
        x[:, :, i] *= std[i]
        x[:, :, i] += mean[i]
    CNT_dict['raw'] = x

CNT_dict = {}
def visulize(mask, raw_feature_map, interp_feature_map, name=''):
    from matplotlib import pyplot as plt
    mask = mask.cpu().numpy()
    raw_feature_map = raw_feature_map.cpu().numpy()
    raw_feature_map_norm = (raw_feature_map[0] ** 2).mean(axis=0)
    interp_feature_map = interp_feature_map.cpu().numpy()
    interp_feature_map_norm = (interp_feature_map[0] ** 2).mean(axis=0)

    if name in CNT_dict:
        CNT_dict[name] = CNT_dict[name] + 1
    else:
        CNT_dict[name] = 0
    plt.subplot(2, 3, 1)
    plt.imshow(CNT_dict['raw'])
    plt.subplot(2, 3, 2)
    plt.imshow(mask[0, 0] > 0, vmin=0, vmax=1)
    plt.subplot(2, 3, 3)
    plt.imshow(mask[0, 0], vmin=0, vmax=1)
    plt.subplot(2, 3, 4)
    plt.imshow(raw_feature_map_norm)
    plt.subplot(2, 3, 5)
    plt.imshow(interp_feature_map_norm)

    plt.savefig('/data/home/zhez/vis/relu/{}_{}.png'.format(name, CNT_dict[name]))

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 normalize=dict(type='BN'),
                 dcn=None,
                 develop=None,
                 name=None):
        super(BasicBlock, self).__init__()
        assert dcn is None, "Not implemented yet."
        assert name is not None
        #assert develop is not None

        self.norm1_name, norm1 = build_norm_layer(normalize, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(normalize, planes, postfix=2)

        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = conv3x3(planes, planes)
        self.add_module(self.norm2_name, norm2)

        self.inplanes = inplanes
        self.planes = planes
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        assert not with_cp

        self.name = name
        self.test_iteration = 0
        self.develop = develop

        if self.name not in Sparsity.keys():
            Sparsity[self.name] = dict()
            Sparsity[self.name]['Conv1'] = 1.0
            Sparsity[self.name]['Conv2'] = 1.0

        self.conv1_mask = nn.Conv2d(inplanes, 1, 3, stride, 1)
        normalize_mask = normalize.copy()
        normalize_mask['affine'] = False
        self.conv1_mask_norm_name, conv1_mask_norm = build_norm_layer(normalize_mask, 1, postfix=3)
        self.add_module(self.conv1_mask_norm_name, conv1_mask_norm)
        self.conv1_mask_relu = nn.ReLU()
        self.conv2_mask = nn.Conv2d(planes, 1, 3, stride, 1)
        self.conv2_mask_norm_name, conv2_mask_norm = build_norm_layer(normalize_mask, 1, postfix=4)
        self.add_module(self.conv2_mask_norm_name, conv2_mask_norm)
        self.conv2_mask_relu = nn.ReLU()

        self.conv1_weight = inplanes / (inplanes + planes)
        self.conv2_weight = planes / (inplanes + planes)
        if develop is not None:
            self.loss_weight = develop.backbone_loss_weight

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def conv1_mask_norm(self):
        return getattr(self, self.conv1_mask_norm_name)

    @property
    def conv2_mask_norm(self):
        return getattr(self, self.conv2_mask_norm_name)

    def forward(self, x):
        if self.develop is None:
            return self.forward_without_mask(x)
        else:
            return self.forward_with_mask(x)

    def forward_without_mask(self, x):
        x, loss = x
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out, 0 + loss

    def forward_with_mask(self, x):
        assert len(x) == 2
        x, loss = x
        residual = x
        n, _, h, w = x.shape

        mask_c1_inp = x
        mask_conv1 = self.conv1_mask(mask_c1_inp)
        mask_conv1_bn = self.conv1_mask_norm(mask_conv1)
        mask_conv1_relu = self.conv1_mask_relu(mask_conv1_bn)

        out_conv1 = self.conv1(x)
        out_conv1_masked = out_conv1 * mask_conv1_relu
        out_bn1 = self.norm1(out_conv1_masked)
        out_conv1 = self.relu(out_bn1)

        if VIS:
            visulize(mask_conv1_relu, out_conv1, out_conv1_masked, name = 'conv1')

        mask_c2_inp = out_conv1
        mask_conv2 = self.conv1_mask(mask_c2_inp)
        mask_conv2_bn = self.conv2_mask_norm(mask_conv2)
        mask_conv2_relu = self.conv2_mask_relu(mask_conv2_bn)

        out_conv2 = self.conv2(out_conv1)
        out_conv2_masked = out_conv2 * mask_conv2_relu
        out_conv2 = self.norm2(out_conv2_masked)
        if VIS:
            visulize(mask_conv2_relu, out_conv2, out_conv2_masked, name = 'conv2')

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out_conv2 + residual
        out = self.relu(out)

        if not self.training:
            mask_conv1_stat = (mask_conv1_relu >= 1e-5).float()
            mask_conv2_stat = (mask_conv2_relu >= 1e-5).float()

            Sparsity[self.name]['Conv1'] = (Sparsity[self.name]['Conv1'] * self.test_iteration + float(
                mask_conv1_stat.mean())) / (self.test_iteration + 1)
            Sparsity[self.name]['Conv2'] = (Sparsity[self.name]['Conv2'] * self.test_iteration + float(
                mask_conv2_stat.mean())) / (self.test_iteration + 1)
            self.test_iteration += 1

        return out, ((mask_conv1_relu > 1e-5).float().mean() * self.conv1_weight + (mask_conv2_relu > 1e-5).float().mean() * self.conv2_weight) * self.loss_weight + loss

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 normalize=dict(type='BN'),
                 dcn=None,
                 develop=None,
                 name=None):
        assert name is not None
        assert develop is not None

        super(Bottleneck, self).__init__()
        assert style in ['pytorch']
        assert dcn is None
        self.inplanes = inplanes
        self.planes = planes
        self.normalize = normalize
        self.stride = stride

        self.norm1_name, norm1 = build_norm_layer(normalize, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(normalize, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(normalize, planes * self.expansion, postfix=3)

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=self.stride, padding=dilation, dilation=dilation, bias=False)
        self.add_module(self.norm2_name, norm2)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp
        self.normalize = normalize

        self.name = name
        self.test_iteration = 0
        self.develop = develop

        if develop.grid_conv:
            mask_planes = inplanes + planes
        else:
            mask_planes = inplanes

        mask_kernel = develop.get('mask_kernel', 3)
        mask_prob = develop.get('mask_prob', 0.9526)
        mask_bias = math.log(mask_prob / (1 - mask_prob))

        self.sigmoid = nn.Sigmoid()

        self.conv1_mask = nn.Conv2d(mask_planes, 1, mask_kernel, 1, mask_kernel // 2)
        constant_init(self.conv1_mask, val=0, bias=mask_bias)
        self.conv1_gumbel = GumbelSigmoid(**develop.gumbel)

        self.conv2_mask = nn.Conv2d(mask_planes, 1, mask_kernel, stride, mask_kernel // 2)
        constant_init(self.conv2_mask, val=0, bias=mask_bias)
        self.conv2_gumbel = GumbelSigmoid(**develop.gumbel)

        self.conv1_sigma = nn.Parameter(torch.tensor(develop.lg_init_sigma))
        self.conv2_sigma = nn.Parameter(torch.tensor(develop.lg_init_sigma))
        self.lg_kernel = develop.lg_kernel
        self.lg_padding = self.lg_kernel // 2
        self.Gaussian = GaussianKernel(self.lg_kernel, develop.lg_type)

        self.grid_conv = develop.grid_conv
        self.grid_mask = develop.grid_mask
        self.grid_interval = develop.grid_interval
        self.mask_type = develop.mask_type
        self.loss_weight = develop.backbone_loss_weight
        assert self.mask_type in ['hard_inference']
        if 'hard' in self.mask_type:
            self.mask_hard_threshold = develop.mask_hard_threshold


    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):

        def _inner_forward(x):
            assert len(x) == 2
            x, loss = x
            residual = x
            n, _, h, w = x.shape

            out_conv1 = self.conv1(x)
            out_bn1 = self.norm1(out_conv1)
            out_conv1 = self.relu(out_bn1)

            if self.grid_conv:
                _conv1_interval = self.grid_interval
                out_grid_conv1 = out_conv1[:, :, ::self.grid_interval, ::self.grid_interval]
                h_res_c1 = (h - 1) % _conv1_interval
                w_res_c1 = (w - 1) % _conv1_interval
                out_grid_conv1 = F.interpolate(out_grid_conv1, size=(h - h_res_c1, w - w_res_c1), mode='bilinear', align_corners=True)
                out_grid_conv1 = F.pad(out_grid_conv1, pad=[0, w_res_c1, 0, h_res_c1], mode='replicate')
                mask_c1_inp = torch.cat(tensors=(x, out_grid_conv1), dim=1)
            else:
                mask_c1_inp = x

            mask_conv1 = self.conv1_mask(mask_c1_inp)
            mask_conv1_sigmoid = self.sigmoid(mask_conv1)
            mask_conv1 = self.conv1_gumbel(mask_conv1_sigmoid)
            if self.grid_mask:
                mask_conv1[:, :, ::self.grid_interval, ::self.grid_interval] = 1.0
            if not self.training and 'hard' in self.mask_type:
                mask_conv1 = mask_conv1 * (mask_conv1 >= self.mask_hard_threshold).float()
                mask_conv1_stat = (mask_conv1 >= self.mask_hard_threshold).float()

            _input_weight_conv1 = self.Gaussian(self.conv1_sigma).reshape((1, 1, self.lg_kernel, self.lg_kernel)).cuda()
            norm_mask_conv1 = F.conv2d(input=mask_conv1, weight=_input_weight_conv1, stride=1, padding=self.lg_padding, dilation=1)
            norm_mask_conv1 = norm_mask_conv1 + 1e-5

            _input_weight_conv1 = _input_weight_conv1.repeat((self.planes, 1, 1, 1))
            out_conv1_sample = out_conv1 * mask_conv1
            out_conv1_interpolate = F.conv2d(input=out_conv1_sample, weight=_input_weight_conv1, stride=1, padding=self.lg_padding, dilation=1, groups=self.planes)
            out_conv1_interpolate = out_conv1_interpolate / norm_mask_conv1
            out_conv1 = mask_conv1 * out_conv1 + (1.0 - mask_conv1) * out_conv1_interpolate

            out_conv2 = self.conv2(out_conv1)
            out_bn2 = self.norm2(out_conv2)
            out_conv2 = self.relu(out_bn2)

            if self.grid_conv:
                _conv2_interval = self.stride * self.grid_interval
                out_grid_conv2 = out_conv2[:, :, ::self.grid_interval, ::self.grid_interval]
                h_res_c2 = (h - 1) % _conv2_interval
                w_res_c2 = (w - 1) % _conv2_interval
                out_grid_conv2 = F.interpolate(out_grid_conv2, size=(h - h_res_c2, w - w_res_c2), mode='bilinear', align_corners=True)
                out_grid_conv2 = F.pad(out_grid_conv2, pad=[0, w_res_c2, 0, h_res_c2], mode='replicate')
                mask_c2_inp = torch.cat(tensors=(x, out_grid_conv2), dim=1)
            else:
                mask_c2_inp = x

            mask_conv2 = self.conv2_mask(mask_c2_inp)
            mask_conv2_sigmoid = self.sigmoid(mask_conv2)
            mask_conv2 = self.conv2_gumbel(mask_conv2_sigmoid)
            if self.grid_mask:
                mask_conv2[:, :, ::self.grid_interval, ::self.grid_interval] = 1.0
            if not self.training and 'hard' in self.mask_type:
                mask_conv2 = mask_conv2 * (mask_conv2 >= self.mask_hard_threshold).float()
                mask_conv2_stat = (mask_conv2 >= self.mask_hard_threshold).float()

            _input_weight_conv2 = self.Gaussian(self.conv2_sigma).reshape((1, 1, self.lg_kernel, self.lg_kernel)).cuda()
            norm_mask_conv2 = F.conv2d(input=mask_conv2, weight=_input_weight_conv2, stride=1, padding=self.lg_padding, dilation=1)
            norm_mask_conv2 = norm_mask_conv2 + 1e-5

            _input_weight_conv2 = _input_weight_conv2.repeat((self.planes, 1, 1, 1))
            out_conv2_sample = out_conv2 * mask_conv2
            out_conv2_interpolate = F.conv2d(input=out_conv2_sample, weight=_input_weight_conv2, stride=1, padding=self.lg_padding, dilation=1, groups=self.planes)
            out_conv2_interpolate = out_conv2_interpolate / norm_mask_conv2
            out_conv2 = mask_conv2 * out_conv2 + (1.0 - mask_conv2) * out_conv2_interpolate

            out_conv3 = self.conv3(out_conv2)
            out_bn3 = self.norm3(out_conv3)

            if self.downsample is not None:
                residual = self.downsample(x)
            out = out_bn3 + residual

            if not self.training:
                Sparsity[self.name]['Conv1'] = (Sparsity[self.name]['Conv1'] * self.test_iteration + float(mask_conv1_stat.mean())) / (self.test_iteration + 1)
                Sparsity[self.name]['Conv2'] = (Sparsity[self.name]['Conv2'] * self.test_iteration + float(mask_conv2_stat.mean())) / (self.test_iteration + 1)
                self.test_iteration += 1

            return out, loss + (0.25 * mask_conv1_sigmoid.mean() + 0.75 * mask_conv2_sigmoid.mean()) * self.loss_weight

        if self.with_cp and x.requires_grad:
            out, loss = cp.checkpoint(_inner_forward, x)
        else:
            out, loss = _inner_forward(x)

        out = self.relu(out)

        return out, loss


def make_res_layer(block,
                   inplanes,
                   planes,
                   blocks,
                   stride=1,
                   dilation=1,
                   style='pytorch',
                   with_cp=False,
                   normalize=dict(type='BN'),
                   dcn=None,
                   develop=None,
                   name=None):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False),
            build_norm_layer(normalize, planes * block.expansion)[1],
        )

    layers = []

    skip_first = develop.get('skip_first', False)

    layers.append(
        block(
            inplanes,
            planes,
            stride,
            dilation,
            downsample,
            style=style,
            with_cp=with_cp,
            normalize=normalize,
            dcn=dcn,
            develop=None if skip_first else develop,
            name=name + f'_Block{1}'
        )
    )

    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(
                inplanes,
                planes,
                1,
                dilation,
                style=style,
                with_cp=with_cp,
                normalize=normalize,
                dcn=dcn,
                develop=develop,
                name=name + f'_Block{i + 1}'
            )
        )

    return nn.Sequential(*layers)


@BACKBONES.register_module
class LCCLNet(nn.Module):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        normalize (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 frozen_stages=-1,
                 normalize=dict(type='BN', frozen=False),
                 norm_eval=False,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 with_cp=False,
                 zero_init_residual=True,
                 develop=None):
        super(LCCLNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for scalenet'.format(depth))
        self.depth = depth
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.frozen_stages = frozen_stages
        self.normalize = normalize
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 64
        self.develop = develop
        self._make_stem_layer()

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            planes = 64 * 2**i
            res_layer = make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                with_cp=with_cp,
                normalize=normalize,
                dcn=dcn,
                develop=develop,
                name=f'Res{i + 1}'
            )
            self.inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * 64 * 2**(
            len(self.stage_blocks) - 1)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self):
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1_name, norm1 = build_norm_layer(
            self.normalize, 64, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for m in [self.conv1, self.norm1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    if m.affine:
                        constant_init(m, 1)

            # for m in self.modules():
            #     if isinstance(m, (BasicBlock, Bottleneck)) and hasattr(m, 'conv1_mask'):
            #         constant_init(m.conv1_mask, 0, 3)
            #     if isinstance(m, (BasicBlock, Bottleneck)) and hasattr(m, 'conv2_mask'):
            #         constant_init(m.conv2_mask, 0, 3)

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(
                            m, 'conv2_offset'):
                        constant_init(m.conv2_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        if self.develop.get('skip_first', False):
            divider = sum(self.stage_blocks) - len(self.stage_blocks)
        else:
            divider = sum(self.stage_blocks)

        if VIS:
            vis_raw_im(x)


        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        loss_backbone = 0.0
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x, loss_backbone = res_layer((x, loss_backbone))
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0], loss_backbone.detach() / divider
        else:
            raise NotImplementedError

    def train(self, mode=True):
        super(LCCLNet, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()