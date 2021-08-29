import logging

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from select_features import SelectFeature
# from ..registry import BACKBONES
from utils import build_norm_layer, GaussianKernel, GumbelSigmoid

# from mmcls.utils import is_host, Sparsity
Sparsity = dict()
torch.set_num_threads(1)
VIS = False

CNT_dict = {}


t_mask_gen = []
def params_count(model):
    return np.sum([p.numel() for p in model.parameters()]).item()
def vis_raw_im(x):
    from matplotlib import pyplot as plt
    x = x[0].cpu().numpy().transpose(1,2,0)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for i in range(3):
        x[:, :, i] *= std[i]
        x[:, :, i] += mean[i]
    # if 'raw' in CNT_dict:
    #     CNT_dict['raw'] += 1
    # else:
    #     CNT_dict['raw'] = 0
    # plt.figure('raw')
    # plt.imshow(x)
    # plt.savefig('/data/home/zhez/vis/ours/raw_{}.png'.format(CNT_dict['raw']))
    CNT_dict['raw'] = x

def visulize(sigmoid, mask, raw_feature_map, interp_feature_map, fusion_map, name=''):
    from matplotlib import pyplot as plt
    sigmoid = sigmoid.cpu().numpy()
    mask = mask.cpu().numpy()
    raw_feature_map = raw_feature_map.cpu().numpy()
    raw_feature_map_norm = (raw_feature_map[0] ** 2).mean(axis=0)
    interp_feature_map = interp_feature_map.cpu().numpy()
    interp_feature_map_norm = (interp_feature_map[0] ** 2).mean(axis=0)
    fusion_map = fusion_map.cpu().numpy()
    fusion_map_norm = (fusion_map[0] ** 2).mean(axis=0)

    if name in CNT_dict:
        CNT_dict[name] = CNT_dict[name] + 1
    else:
        CNT_dict[name] = 0
    plt.figure(name)
    print(CNT_dict[name])
    #plt.figure("{0}_{1}".format(name, CNT_dict[name]))
    plt.subplot(2, 3, 1)
    plt.imshow(CNT_dict['raw'])
    plt.subplot(2, 3, 2)
    plt.imshow(sigmoid[0, 0], vmin=0, vmax=1)
    plt.subplot(2, 3, 3)
    plt.imshow(mask[0, 0], vmin=0, vmax=1)
    plt.subplot(2, 3, 4)
    plt.imshow(raw_feature_map_norm)
    plt.subplot(2, 3, 5)
    plt.imshow(interp_feature_map_norm)
    plt.savefig('/data/home/zhez/vis/ours/{}_{}.png'.format(name, CNT_dict[name]))

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

        if self.name not in Sparsity.keys():
            Sparsity[self.name] = dict()
            Sparsity[self.name]['Conv1'] = 1.0
            Sparsity[self.name]['Conv2'] = 1.0

        self.develop = develop
        if self.develop is not None:
            if develop.get('grid_conv'):
                mask_planes = inplanes + planes
            else:
                mask_planes = inplanes

            mask_kernel = develop.get('mask_kernel', 3)
            mask_prob = develop.get('mask_prob', 0.9526)
            mask_bias = math.log(mask_prob / (1 - mask_prob))

            self.sigmoid = nn.Sigmoid()

            self.conv1_mask = nn.Conv2d(mask_planes, 1, mask_kernel, stride, mask_kernel // 2)
            constant_init(self.conv1_mask, val=0, bias=mask_bias)
            self.conv1_gumbel = GumbelSigmoid(**develop.get('gumbel'))

            self.conv2_mask = nn.Conv2d(mask_planes, 1, mask_kernel, stride, mask_kernel // 2)
            constant_init(self.conv2_mask, val=0, bias=mask_bias)
            self.conv2_gumbel = GumbelSigmoid(**develop.get('gumbel'))

            
            self.lg_init_sigma = develop.get('lg_init_sigma')
            _lg_sigma_c1 = torch.tensor(self.lg_init_sigma)
            _lg_sigma = torch.tensor(self.lg_init_sigma)
            self.lg_sigma_c1 = nn.Parameter(_lg_sigma_c1)
            self.lg_sigma = nn.Parameter(_lg_sigma)
            self.lg_kernel = develop.get('lg_kernel')
            self.lg_padding = (self.lg_kernel - 1) // 2
            
            
            self.conv1_sigma = nn.Parameter(torch.tensor(develop.get('lg_init_sigma')))
            self.conv2_sigma = nn.Parameter(torch.tensor(develop.get('lg_init_sigma')))

            self.lg_kernel = develop.get('lg_kernel')
            # self.lg_padding = self.lg_kernel // 2
            self.Gaussian = GaussianKernel(self.lg_kernel, develop.get('lg_type'))

            self.grid_conv = develop.get('grid_conv')
            self.grid_mask = develop.get('grid_mask')
            self.grid_interval = develop.get('grid_interval')
            self.mask_type = develop.get('mask_type')
            self.loss_weight = develop.get('backbone_loss_weight')
            self.loss_position = develop.get('loss_position', 'sigmoid')
            assert self.mask_type in ['hard_inference']
            if 'hard' in self.mask_type:
                self.mask_hard_threshold = develop.get('mask_hard_threshold')

            self.conv1_weight = inplanes / (inplanes + planes)
            self.conv2_weight = planes / (inplanes + planes)
            self._input_weight = dict()
    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)
    
    
    
    def apply_proxy_dcn_bn_relu(self, input, conv, offset=None, mask=None, mask_loc=None, bn=None, relu=None, sparse_output=False, ones_mode=False, depth_conv=False):
        # global conv_im2col_time, conv_remove_time, conv_init_time, conv_output_time, conv_computation_time

        # cur_remove_time = 0.0
        t = time.time()
        if ones_mode:
            init_F = torch.ones
        else:
            init_F =torch.zeros

        oc, ic, ks, _ = conv.weight.shape
        _, _, oh, ow = self.get_out_shape(input, conv)
        x = torch.nn.functional.pad(input, [0, 1-input.shape[3]%2, 0, 1-input.shape[2]%2], mode='constant', value=0)
        n, c, h, w = x.shape

        if mask_loc is None:
            n_mask = oh*ow
        else:
            n_mask = mask_loc.shape[0]
            if n_mask == 0:
                if sparse_output:
                    return init_F((n, oc, n_mask), device=x.device)
                else:
                    return init_F((n, oc, oh, ow), device=x.device)

        if offset is None:
            offset = torch.zeros((n, ks * ks * 2, n_mask), device=x.device).float()
            use_select_feature = True
        else:
            use_select_feature = False


        kd = torch.arange(ks, device=x.device).float()
        ky, kx = torch.meshgrid(kd, kd)

        if mask_loc is not None:
            dx = mask_loc[:, 1].reshape((1, 1, 1, n_mask))
            dy = mask_loc[:, 0].reshape((1, 1, 1, n_mask))
        else:
            dx = torch.arange(ow, device=x.device).float()
            dy = torch.arange(oh, device=x.device).float()
            dy, dx = torch.meshgrid(dy, dx)
            dx = dx.reshape((1, 1, 1, oh*ow))
            dy = dy.reshape((1, 1, 1, oh*ow))

            if mask is not None:
                dx = dx[:, :, :, mask]
                dy = dy[:, :, :, mask]
        dx = dx * conv.stride[0] - conv.padding[0]
        dy = dy * conv.stride[0] - conv.padding[0]

        offset = offset.view((-1, ks * ks, 2, n_mask))

        dx = dx + kx.reshape((1, 1, ks*ks, 1))
        dy = dy + ky.reshape((1, 1, ks*ks, 1))

        offset[:, :, 0, :] = offset[:, :, 0, :] + dy
        offset[:, :, 1, :] = offset[:, :, 1, :] + dx

        if use_select_feature:
            # cur_remove_time += time.time() - t
            t = time.time()
            x = SelectFeature(x, offset.permute((0, 1, 3, 2)))
            # conv_im2col_time += time.time() - t
        else:
            offset[:,:,0,:] = offset[:,:,0,:] / (h - 1) * 2 - 1.0
            offset[:,:,1,:] = offset[:,:,1,:] / (w - 1) * 2 - 1.0
            offset = offset.permute((0, 1, 3, 2)).flip(dims=[3])
            offset_repeat = offset.repeat((c, 1, 1, 1))
            # cur_remove_time += time.time() - t

            t = time.time()
            x = nn.functional.grid_sample(x.view(-1, 1, h, w), offset_repeat)
            # conv_im2col_time += time.time() - t

        t = time.time()
        if depth_conv:
            x = x.view((n, c, ks*ks, -1))
            weight_flat = conv.weight.data.view(1, oc, -1, 1)
            x = (x * weight_flat).sum(2, keepdims=True)
        else:
            assert n==1
            # x = x.view((n, c * ks * ks, -1))
            # weight_flat = conv.weight.data.view(oc, -1)
            # y = (weight_flat @ x)[:, :, None, :]

            x = x.view((c * ks * ks, -1))
            weight_flat = conv.weight.data.view(oc, -1)
            x = torch.matmul(weight_flat, x)[None, :, None, :]

        # conv_computation_time += time.time() - t
        if type(conv.bias) is bool:
            if conv.bias:
                x = x + conv.bias.view((1, -1, 1, 1))
        elif conv.bias is not None:
            x = x + conv.bias.view((1, -1, 1, 1))

        if bn is not None:
            x = bn(x)
        if relu is not None:
            x = relu(x)

        t = time.time()
        if sparse_output:
            res = x.view((-1, oc, n_mask))
        else:
            if mask is not None:
                if n*c*h*w == n*oc*oh*ow:
                    if ones_mode:
                        out = input.reshape((n, oc, 1, oh*ow)) * 0 + 1
                    else:
                        out = input.reshape((n, oc, 1, oh*ow)) * 0
                else:
                    out = init_F((n, oc, 1, oh*ow), device=x.device)
                out[:,:,:, mask] = x

                # out[0, mask, :] = x[0,:,0,:].T
                # out = out.permute((0,2,1))
            else:
                out = x
            
            res = out.view((-1, oc, oh, ow))
        # conv_output_time += time.time() - t
        # conv_remove_time += cur_remove_time

        return res, (n, oc, oh, ow)
    
    def get_out_shape(self, x, conv):
        b, c, h, w = x.shape
        padding = conv.padding[0]
        dilation = conv.dilation[0]
        oc, ic, ks, _ = conv.weight.shape
        stride = conv.stride[0]
       
        oh = (h + 2 * padding - dilation * (ks-1) - 1) // stride + 1
        ow = (w + 2 * padding - dilation * (ks-1) - 1) // stride + 1
        
        return b, oc, oh, ow
    
    def forward(self, x, sparse=False):
        if self.develop is None:
            return self.forward_without_mask(x)
        else:
            return self.forward_with_mask(x, sparse=sparse)

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

    def forward_with_mask(self, x, sparse=False):
        assert len(x) == 2
        x, loss = x
        residual = x
        n, _, h, w = x.shape


        # if self.grid_conv:
        #     _conv1_interval = self.stride * self.grid_interval
        #     out_grid_conv1 = out_conv1[:, :, ::self.grid_interval, ::self.grid_interval]
        #     h_res_c1 = (h - 1) % _conv1_interval
        #     w_res_c1 = (w - 1) % _conv1_interval
        #     out_grid_conv1 = F.interpolate(out_grid_conv1, size=(h - h_res_c1, w - w_res_c1), mode='bilinear', align_corners=True)
        #     out_grid_conv1 = F.pad(out_grid_conv1, pad=[0, w_res_c1, 0, h_res_c1], mode='replicate')
        #     mask_c1_inp = torch.cat(tensors=(x, out_grid_conv1), dim=1)
        # else:
        mask_c1_inp = x

        mask_conv1 = self.conv1_mask(mask_c1_inp)
        mask_conv1_sigmoid = self.sigmoid(mask_conv1)
        mask_conv1 = self.conv1_gumbel(mask_conv1_sigmoid)
        if self.grid_mask:
            mask_conv1[:, :, ::self.grid_interval, ::self.grid_interval] = mask_conv1[:, :, ::self.grid_interval, ::self.grid_interval] + 1
            mask_conv1 = torch.clamp(mask_conv1, max=1.0)
        if not self.training and 'hard' in self.mask_type:
            mask_conv1 = mask_conv1 * (mask_conv1 >= self.mask_hard_threshold).float()
            mask_conv1_stat = (mask_conv1 >= self.mask_hard_threshold).float()
        _input_weight_conv1 = self.Gaussian(self.conv1_sigma).reshape((1, 1, self.lg_kernel, self.lg_kernel))
        norm_mask_conv1 = F.conv2d(input=mask_conv1, weight=_input_weight_conv1, stride=1, padding=self.lg_padding, dilation=1)
        norm_mask_conv1 = norm_mask_conv1 + 1e-5

        _input_weight_conv1 = _input_weight_conv1.repeat((self.planes, 1, 1, 1))
        
        # mask_conv1 = torch.ones(mask_conv1.shape)
        # mask_conv1[:,:,::2,:] = 0.0
        t0 = time.time()
        mask_conv1 = torch.zeros(mask_conv1.shape, device=x.device)
        mask_conv1[:,:,::2,:] = 1.0
        t1 = time.time()
        t_mask_gen.append(t1-t0)
        # print(mask_conv1.sum()/mask_conv1.numel())
        
        if not sparse:
            # print('not sparse')
            out_conv1 = self.conv1(x)
            out_bn1 = self.norm1(out_conv1)
            out_conv1 = self.relu(out_bn1)
            # print(x.shape, mask_conv1.shape)
            out_conv1_sample = out_conv1 * mask_conv1
            out_conv1_interpolate = F.conv2d(input=out_conv1_sample, weight=_input_weight_conv1, stride=1, padding=self.lg_padding, dilation=1,
                                        groups=self.planes)
            out_conv1_interpolate = out_conv1_interpolate / norm_mask_conv1
            out_conv1_fusion = mask_conv1 * out_conv1 + (1.0 - mask_conv1) * out_conv1_interpolate
        
        else:
            # print('sparse')
            binary_mask_ = (mask_conv1 > 0)
            # binary_mask_ = (mask_conv1 > self.mask_hard_threshold)
            binary_mask_flatten = binary_mask_.flatten()
            loc_mask_ = binary_mask_.nonzero()[:, 2:].float()
            out_conv1_sparse, (ob, oc, oh, ow) = self.apply_proxy_dcn_bn_relu(x, self.conv1, offset=None,
                                                        mask=binary_mask_flatten, bn=self.norm1,
                                                        relu=self.relu, sparse_output=True, mask_loc=loc_mask_)

            #  = self.get_out_shape(x, self.conv1)
            out = torch.zeros((ob, oc, oh * ow), device=x.device)
            # print(out[:,:,binary_mask_flatten].shape, out_conv1_sparse.shape)
            out[:,:,binary_mask_flatten] = out_conv1_sparse
            out_conv1_sparse = out.view(ob, oc, oh, ow)
            out_conv1_interpolate_sparse = F.conv2d(input=out_conv1_sparse, weight=_input_weight_conv1, stride=1, padding=self.lg_padding, dilation=1,
                                            groups=self.planes)
            out_conv1_interpolate_sparse = out_conv1_interpolate_sparse / norm_mask_conv1
            out_conv1_fusion_sparse = mask_conv1 * out_conv1_sparse + (1.0 - mask_conv1) * out_conv1_interpolate_sparse
        
            out_conv1_fusion = out_conv1_fusion_sparse
        
        
        mask_c2_inp = x
        mask_conv2 = self.conv2_mask(mask_c2_inp)
        mask_conv2_sigmoid = self.sigmoid(mask_conv2)
        mask_conv2 = self.conv2_gumbel(mask_conv2_sigmoid)
        if self.grid_mask:
            mask_conv2[:, :, ::self.grid_interval, ::self.grid_interval] = mask_conv2[:, :, ::self.grid_interval, ::self.grid_interval] + 1
            mask_conv2 = torch.clamp(mask_conv2, max=1.0)

        if not self.training and 'hard' in self.mask_type:
            mask_conv2 = mask_conv2 * (mask_conv2 >= self.mask_hard_threshold).float()
            mask_conv2_stat = (mask_conv2 >= self.mask_hard_threshold).float()

        _input_weight_conv2 = self.Gaussian(self.conv2_sigma).reshape((1, 1, self.lg_kernel, self.lg_kernel))
        norm_mask_conv2 = F.conv2d(input=mask_conv2, weight=_input_weight_conv2, stride=1, padding=self.lg_padding, dilation=1)
        norm_mask_conv2 = norm_mask_conv2 + 1e-5

        _input_weight_conv2 = _input_weight_conv2.repeat((self.planes, 1, 1, 1))
        
        t0 = time.time()
        mask_conv2 = torch.zeros(mask_conv2.shape, device=x.device)
        mask_conv2[:,:,::2,:] = 1.0
        t1 = time.time()
        t_mask_gen.append(t1-t0)
        # print(mask_conv2.sum()/mask_conv2.numel())
        # assert(0==1)
        # mask_conv2 = torch.ones(mask_conv2.shape)
        # mask_conv2[:,:,::2,:] = 0.0
        
        
        if not sparse:
            # print('not sparse')
            out_conv2 = self.conv2(out_conv1_fusion)
            out_bn2 = self.norm2(out_conv2)
            out_conv2 = self.relu(out_bn2)
            # print(x.shape, mask_conv1.shape)
            out_conv2_sample = out_conv2 * mask_conv2
            out_conv2_interpolate = F.conv2d(input=out_conv2_sample, weight=_input_weight_conv2, stride=1, padding=self.lg_padding, dilation=1,
                                        groups=self.planes)
            out_conv2_interpolate = out_conv2_interpolate / norm_mask_conv2
            out_conv2_fusion = mask_conv2 * out_conv2 + (1.0 - mask_conv2) * out_conv2_interpolate
        
        
        else:
            # print('sparse')
            binary_mask_ = (mask_conv2 > 0)
            # binary_mask_ = (mask_conv1 > self.mask_hard_threshold)
            binary_mask_flatten = binary_mask_.flatten()
            loc_mask_ = binary_mask_.nonzero()[:, 2:].float()
            out_conv2_sparse, (ob, oc, oh, ow) = self.apply_proxy_dcn_bn_relu(out_conv1_fusion, self.conv2, offset=None,
                                                        mask=binary_mask_flatten, bn=self.norm1,
                                                        relu=self.relu, sparse_output=True, mask_loc=loc_mask_)

            
            #  = self.get_out_shape(x, self.conv1)
            out = torch.zeros((ob, oc, oh * ow), device=x.device)
            # print(out[:,:,binary_mask_flatten].shape, out_conv1_sparse.shape)
            out[:,:,binary_mask_flatten] = out_conv2_sparse
            out_conv2_sparse = out.view(ob, oc, oh, ow)
            out_conv2_interpolate_sparse = F.conv2d(input=out_conv2_sparse, weight=_input_weight_conv2, stride=1, padding=self.lg_padding, dilation=1,
                                            groups=self.planes)
            out_conv2_interpolate_sparse = out_conv2_interpolate_sparse / norm_mask_conv2
            out_conv2_fusion_sparse = mask_conv2 * out_conv2_sparse + (1.0 - mask_conv2) * out_conv2_interpolate_sparse
            out_conv2_fusion = out_conv2_fusion_sparse
            
            # print((out_conv2_fusion - out_conv2_fusion_sparse).abs().sum())
        
            

        # if VIS:
        #     visulize(mask_conv2_sigmoid, mask_conv2, out_conv2, out_conv2_interpolate, out_conv2_fusion, name = 'conv2')

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out_conv2_fusion + residual
        out = self.relu(out)

        if not self.training:
            # Sparsity[self.name]['Conv1'] = (Sparsity[self.name]['Conv1'] * self.test_iteration + float(
            #     mask_conv1_stat.mean())) / (self.test_iteration + 1)
            # Sparsity[self.name]['Conv2'] = (Sparsity[self.name]['Conv2'] * self.test_iteration + float(
            #     mask_conv2_stat.mean())) / (self.test_iteration + 1)
            
            Sparsity[self.name]['Conv1'] = (Sparsity[self.name]['Conv1'] * self.test_iteration + float(
                mask_conv1.mean())) / (self.test_iteration + 1)
            Sparsity[self.name]['Conv2'] = (Sparsity[self.name]['Conv2'] * self.test_iteration + float(
                mask_conv2.mean())) / (self.test_iteration + 1)
            self.test_iteration += 1

        if self.loss_position == 'sigmoid':
            return out, (mask_conv1_sigmoid.mean() * self.conv1_weight + mask_conv2_sigmoid.mean() * self.conv2_weight) * self.loss_weight + loss
        elif self.loss_position == 'gumble_sigmoid':
            return out, (mask_conv1.mean() * self.conv1_weight + mask_conv2.mean() * self.conv2_weight) * self.loss_weight + loss

        
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

            _input_weight_conv1 = self.Gaussian(self.conv1_sigma).reshape((1, 1, self.lg_kernel, self.lg_kernel))
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

            _input_weight_conv2 = self.Gaussian(self.conv2_sigma).reshape((1, 1, self.lg_kernel, self.lg_kernel))
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

    return nn.ModuleList(layers)


# @BACKBONES.register_module
class ScaleNet(nn.Module):
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
        super(ScaleNet, self).__init__()
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
        self.init_mask_bias = self.develop.get('init_mask_bias', 3)
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
                    constant_init(m, 1)

            for m in self.modules():
                if isinstance(m, (BasicBlock, Bottleneck)) and hasattr(m, 'conv1_mask'):
                    constant_init(m.conv1_mask, 0, self.init_mask_bias)
                if isinstance(m, (BasicBlock, Bottleneck)) and hasattr(m, 'conv2_mask'):
                    constant_init(m.conv2_mask, 0, self.init_mask_bias)

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

    def forward(self, x, sparse=False):
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
            # print(res_layer)
            # assert(0==1)
            for i in range(len(res_layer)):
                x, loss_backbone = res_layer[i]((x, loss_backbone), sparse=sparse)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0], loss_backbone / divider
        else:
            raise NotImplementedError

    def train(self, mode=True):
        super(ScaleNet, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()



if __name__ == "__main__":
    import argparse
    import time
    import numpy as np
    t_mask_gen_avg = []
    model_34_cls = ScaleNet(depth=34,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        develop=dict(
            gumbel=dict(
                decay_method='exp',
                max_T=1.0,
                decay_alpha=0.999988,
            ),
            mask_kernel=3,
            mask_prob=0.9526,
            mask_type='hard_inference',
            mask_hard_threshold=0.5,
            grid_conv=False,
            grid_mask=True,
            grid_interval=2,
            interpolate='lg',
            lg_kernel=5,
            lg_init_sigma=3.0,
            lg_type='reciprocal',
            backbone_loss_weight=0.05,
        ))

    x = torch.rand(1, 3, 224, 224).cuda()
    model_34_cls = model_34_cls.cuda()
    print(params_count(model_34_cls)/ 1e6)
    print('cuda ok')
    with torch.no_grad():
        
        model_34_cls.eval()
        
        # y1, _ = model_34_cls(x, sparse=False)
        # y2, _ = model_34_cls(x, sparse=True)
        # exit()
        # print((y1-y2).abs().sum())
        a=[]
        for i in range(120):
            t = time.time()
            _, _ = model_34_cls(x, sparse=True)

            tt = time.time()
            print((tt - t) * 1000)
            if i >= 20:
                a.append((tt - t) * 1000)
                t_mask_gen_avg.append(np.sum(t_mask_gen)*1000)
                t_mask_gen = []

        print('avg_time', np.mean(a),np.std(a))
        print('avg_time_mask_gen', np.mean(t_mask_gen_avg),np.std(t_mask_gen_avg))
        print(Sparsity)
        