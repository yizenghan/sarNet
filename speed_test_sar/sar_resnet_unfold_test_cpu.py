import torch.nn as nn
# from torch.hub import load_state_dict_from_url
import torch
import torch.nn.functional as F
from gumbel_softmax import GumbleSoftmax
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
# from einops import rearrange, reduce, repeat



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
        residual = x
        if self.downsample is not None:  # skip connection before mask
            residual = self.downsample(x)

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
            if mask.sum() == 0.0:
                out = self.bn3(torch.zeros(residual.shape))
                out += residual

                if self.last_relu:
                    out = self.relu(out)
                return out

            b, c, h, w = x.shape

            g = mask.shape[1]
            m_h = mask.shape[2]
            if g > 1:
                mask1 = mask.unsqueeze(1).repeat(1, c // g, 1, 1, 1).transpose(1, 2).reshape(b, c, m_h, m_h)
            else:
                mask1 = mask.clone()
            mask1 = F.interpolate(mask1, size=(h, w))
            # print(mask1.shape, x.shape)
            x_in = x * mask1
            inter = x_in.shape[2] // mask.shape[2]
            x_in = F.pad(x_in, (1,) * 4)
            # out_ls = []

            for i in range(g):
                if mask[0, i, :, :].sum() == 0:
                    continue
                c_out_g = self.conv1.out_channels // g
                c_in = c // g

                idx = torch.nonzero(mask[:,i,:,:])
                idx = idx[:, 0], idx[:, 1], idx[:, 2]
                patch = x_in[:,i*c_in:(i + 1)*c_in,:,:].permute(0, 2, 3, 1).unfold(1, inter+2, inter).unfold(2, inter+2, inter)[idx[0], idx[1], idx[2]]

                weight = self.conv1.weight
                weight_g = weight[i * c_out_g:(i + 1) * c_out_g, :, :, :]

                out = F.conv2d(patch, weight_g, padding=0)

                # rm = self.bn1.running_mean[i * c_out_g:(i + 1) * c_out_g]
                # rv = self.bn1.running_var[i * c_out_g:(i + 1) * c_out_g]
                # w_bn = self.bn1.weight[i * c_out_g:(i + 1) * c_out_g]
                # b_bn = self.bn1.bias[i * c_out_g:(i + 1) * c_out_g]

                # out = F.batch_norm(out, running_mean=rm, running_var=rv, weight=w_bn, bias=b_bn, training=self.training,
                #                    momentum=0.1, eps=1e-05)
                # out = self.bn1(out)
                out = self.relu(out)

                weight = self.conv2.weight
                c_out_g = self.conv2.out_channels // g
                weight_g = weight[i * c_out_g:(i + 1) * c_out_g, :, :, :]

                out = F.conv2d(out, weight_g, padding=0)

                # rm = self.bn2.running_mean[i * c_out_g:(i + 1) * c_out_g]
                # rv = self.bn2.running_var[i * c_out_g:(i + 1) * c_out_g]
                # w_bn = self.bn2.weight[i * c_out_g:(i + 1) * c_out_g]
                # b_bn = self.bn2.bias[i * c_out_g:(i + 1) * c_out_g]

                # out = F.batch_norm(out, running_mean=rm, running_var=rv, weight=w_bn, bias=b_bn, training=self.training,
                #                    momentum=0.1, eps=1e-05)
                out = self.relu(out)

                weight = self.conv3.weight
                c_out_g = self.conv3.out_channels // g
                weight_g = weight[i * c_out_g:(i + 1) * c_out_g, :, :, :]

                out = F.conv2d(out, weight_g, padding=0)

                # rm = self.bn3.running_mean[i * c_out_g:(i + 1) * c_out_g]
                # rv = self.bn3.running_var[i * c_out_g:(i + 1) * c_out_g]
                # w_bn = self.bn3.weight[i * c_out_g:(i + 1) * c_out_g]
                # b_bn = self.bn3.bias[i * c_out_g:(i + 1) * c_out_g]

                # out = F.batch_norm(out, running_mean=rm, running_var=rv, weight=w_bn, bias=b_bn, training=self.training,
                #                    momentum=0.1, eps=1e-05)

                br,cr,hr,wr = residual.shape
                # x0 = torch.zeros(br,cr//g,hr,wr)
                x0 = torch.zeros(br,cr//g,hr,wr)
                xB, xC, xH, xW = x0.shape
                yB, yC, yH, yW = out.shape
                x0 = x0.view(xB, xC, xH // yH, yH, xW // yW, yW).permute(0, 2, 4, 1, 3, 5)
                x0[idx[0], idx[1], idx[2]] = out
                out = x0.permute(0, 3, 1, 4, 2, 5).view(xB, xC, xH, xW)
                if i == 0:
                    outs = out
                else:
                    outs = torch.cat((outs, out), 1)
                # out_ls.append(out)

            # outs = torch.stack(out_ls, dim=1)
            # outs = out_ls[0]
            # print(outs.shape)
            # for i in range(g-1):
            #     print(out_ls[i+1].shape)
            #     outs = torch.cat((outs, out_ls[i+1]), 1)
            # print(outs.shape, residual.shape)
            assert(outs.shape == residual.shape)
            outs+=residual

            # outs+=residual
            if self.last_relu:
                outs = self.relu(outs)
            return outs


    def forward_calc_flops(self, x, mask, inference=False):
        # print('refine bottleneck, input shape: ', x.shape)
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
        out += residual
        if self.last_relu:
            out = self.relu(out)
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
        _masks = []
        x_refine = x
        # refine_ls = []
        for i in range(len(self.base_module)):
            x_base = self.base_module[i](x_base) if i != 0 else self.base_module[i](x)
            mask = self.mask_gen[i](x_base, temperature=temperature)
            mask = torch.zeros((1, self.patch_groups, self.mask_size, self.mask_size))
            n = int(self.patch_groups / 2)

            # mask[:, :, :, :] = 1.0
            # ### g2 r0.7
            # if len(self.base_module) == 5:
            #     mask[:, :, :, :1] = 1.0
            #     mask[:, n - 1:n, :, :] = 1.0
            # else:
            #     mask[:, :, :5, :] = 1.0
            ### g2 r0.6
            # if len(self.base_module)==5:
            #     mask[:,:,:1,:]=1.0
            #     mask[:,n:n+1,1:,1:]=1.0
            # else:
            #     mask[:,:,:4,:]=1.0
            ### g2 r0.4
            # if len(self.base_module)==5:
            #     mask[:,:,:1,:1]=1.0
            #     mask[:,n-1:n,:1,0:]=1.0
            # else:
            #     mask[:,:,:3,:]=1.0
            ### g2g4 r0.5
            # if len(self.base_module) == 5:
            #     mask[:, :, :, :1] = 1.0
            # else:
            #     mask[:, :n, :3, :] = 1.0
            #     mask[:, n:, 3:, :] = 1.0
            mask[:,0,::2,::2]=1.0
            mask[:,0,1::2,1::2]=1.0
            mask[:,1,::2,::2]=1.0
            mask[:,1,1::2,1::2]=1.0
            # print(mask)
            # print(mask.sum()/mask.numel())
            # assert(0==1)
            # mask[:, 1, : :] = 1.0
            # print(mask)
            ### g4 r0.7
            # if len(self.base_module) == 5:
            #     mask[:, :, :, :1] = 1.0
            #     mask[:, n - 2:n + 1, 1:, 1:] = 1.0
            # else:
            #     mask[:, :, :5, :] = 1.0
            ### g4 r0.6
            # if len(self.base_module) == 5:
            #     mask[:, :, :1, :] = 1.0
            #     mask[:, n - 1:n + 1, 1:, 1:] = 1.0
            # else:
            #     mask[:, :, :4, :] = 1.0
            ### g4 r0.4
            # if len(self.base_module)==5:
            #     mask[:,:,:1,:1]=1.0
            #     mask[:,n-1:n+1,:1,0:]=1.0
            # else:
            #     mask[:,:,:3,:]=1.0


            # ratio = mask.sum() / mask.numel()
            # print('ratio:', ratio)
            _masks.append(mask)
            x_refine = self.refine_module[i](x_refine, mask, inference=inference)
            # refine_ls.append(x_refine)
        if self.alpha != 1:
            x_base = self.base_transform(x_base)
        if self.beta != 1:
            x_refine = self.refine_transform(x_refine)
        _, _, h, w = x_refine.shape
        x_base = F.interpolate(x_base, size=(h, w))
        out = self.relu(x_base + x_refine)
        out = self.fusion[0](out)
        return out, _masks

    def forward_calc_flops(self, x, temperature=1e-8, inference=False):
        b, c, h, w = x.size()
        flops = 0
        _masks = []
        x_refine = x
        # refine_ls = []
        for i in range(len(self.base_module)):
            x_base, _flops = self.base_module[i].forward_calc_flops(x_base) if i != 0 else self.base_module[
                i].forward_calc_flops(x)
            flops += _flops
            mask, _flops = self.mask_gen[i].forward_calc_flops(x_base, temperature=temperature)
            mask = torch.zeros((1, self.patch_groups, self.mask_size, self.mask_size))
            n = int(self.patch_groups / 2)

            # mask[:, :, :, :] = 1.0
            # ### g2 r0.7
            # if len(self.base_module) == 5:
            #     mask[:, :, :, :1] = 1.0
            #     mask[:, n - 1:n, :, :] = 1.0
            # else:
            #     mask[:, :, :5, :] = 1.0
            ### g2 r0.6
            # if len(self.base_module)==5:
            #     mask[:,:,:1,:]=1.0
            #     mask[:,n:n+1,1:,1:]=1.0
            # else:
            #     mask[:,:,:4,:]=1.0
            ### g2 r0.4
            # if len(self.base_module)==5:
            #     mask[:,:,:1,:1]=1.0
            #     mask[:,n-1:n,:1,0:]=1.0
            # else:
            #     mask[:,:,:3,:]=1.0
            ### g2g4 r0.5
            if len(self.base_module) == 5:
                mask[:, :, :, :1] = 1.0
            else:
                mask[:, :n, :3, :] = 1.0
                mask[:, n:, 3:, :] = 1.0
            ### g4 r0.7
            # if len(self.base_module) == 5:
            #     mask[:, :, :, :1] = 1.0
            #     mask[:, n - 2:n + 1, 1:, 1:] = 1.0
            # else:
            #     mask[:, :, :5, :] = 1.0
            ### g4 r0.6
            # if len(self.base_module) == 5:
            #     mask[:, :, :1, :] = 1.0
            #     mask[:, n - 1:n + 1, 1:, 1:] = 1.0
            # else:
            #     mask[:, :, :4, :] = 1.0
            ### g4 r0.4
            # if len(self.base_module)==5:
            #     mask[:,:,:1,:1]=1.0
            #     mask[:,n-1:n+1,:1,0:]=1.0
            # else:
            #     mask[:,:,:3,:]=1.0

            # ratio = mask.sum() / mask.numel()
            # print('ratio:', ratio)
            _masks.append(mask)
            flops += _flops
            x_refine, _flops = self.refine_module[i].forward_calc_flops(x_refine, mask, inference=inference)
            # refine_ls.append(x_refine)
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
        return out, _masks, flops


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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print('before layer 1:', x.shape)
        _masks = []
        x, mask = self.layer1(x, temperature=temperature, inference=inference)
        _masks.extend(mask)

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
        flops += c_in * x.shape[1] * x.shape[2] * x.shape[3] * self.conv1.weight.shape[2] * self.conv1.weight.shape[3]
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)
        flops += x.numel() / x.shape[0] * 9
        _masks = []
        # print(x.shape)
        x, mask, _flops = self.layer1.forward_calc_flops(x, temperature=temperature, inference=inference)
        _masks.extend(mask)
        flops += _flops
        # print(x.shape)
        x, mask, _flops = self.layer2.forward_calc_flops(x, temperature=temperature, inference=inference)
        _masks.extend(mask)
        flops += _flops
        # print(x.shape)
        x, mask, _flops = self.layer3.forward_calc_flops(x, temperature=temperature, inference=inference)
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

        return x, _masks, flops



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


def sar_resnet34_alphaBase_4stage_imgnet(args):
    return sar_resnet_imgnet_alphaBase(depth=34, num_classes=args.num_classes, patch_groups=args.patch_groups,
                                       mask_size=args.mask_size, alpha=args.alpha, beta=args.beta,
                                       base_scale=args.base_scale)


def sar_resnet50_alphaBase_4stage_imgnet(args):
    return sar_resnet_imgnet_alphaBase(depth=50, num_classes=args.num_classes, patch_groups=args.patch_groups,
                                       mask_size=args.mask_size, alpha=args.alpha, beta=args.beta,
                                       base_scale=args.base_scale)


if __name__ == "__main__":
    import argparse
    # from op_counter import measure_model
    import time
    import numpy as np

    parser = argparse.ArgumentParser(description='PyTorch SARNet')
    args = parser.parse_args()
    args.num_classes = 1000
    args.patch_groups = 2
    args.mask_size = 7
    args.alpha = 2
    args.beta = 1
    args.base_scale = 2
    sar_res = sar_resnet50_alphaBase_4stage_imgnet(args)
    # print(sar_res)

    with torch.no_grad():
        # print(sar_res)
        x = torch.rand(1, 3, 224, 224)
        sar_res.eval()
        from conv_bn_fuse import fuse_module
        fuse_module(sar_res)
        b=[]
        g=[]

        # for i in range(10):
        #     _, y2, _masks = sar_res(x,inference=True,temperature=1e-8)

        for i in range(100):
            t = time.time()
            y1, _ = sar_res(x, inference=False, temperature=1e-8)
            tttt = time.time()
            b.append((tttt - t) * 1000)
            print(i, (tttt - t) * 1000)

        for i in range(100):
            tt = time.time()
            y2, _ = sar_res(x,inference=True,temperature=1e-8)
            ttt = time.time()
            g.append((ttt-tt)*1000)
            print(i, (ttt-tt)*1000)

        print('total_time_calc', np.mean(b))
        print('-------------------------------')

        print('total_time', np.mean(g))
        print('-------------------------------')

        y1, _masks1 = sar_res(x,inference=False,temperature=1e-8)
        y2, _masks2 = sar_res(x,inference=True, temperature=1e-8)

        print((y1-y2).abs().sum())
        # print((ls[0]-ls1[0]).abs().sum())

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


