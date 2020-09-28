import torch.nn as nn
# from torch.hub import load_state_dict_from_url
import torch
import torch.nn.functional as F
from gumbel_softmax import GumbleSoftmax
import matplotlib.pyplot as plt 

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1,groups=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,groups=groups)

class Sequential_ext(nn.Module):
    """A Sequential container extended to also propagate the gating information
    that is needed in the target rate loss.
    """

    def __init__(self, *args):
        super(Sequential_ext, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def forward(self, input, temperature=1, openings=None):
        gate_activations = []
        for i, module in enumerate(self._modules.values()):
            input, gate_activation = module(input, temperature)
            
            gate_activations.append(gate_activation)
            # if gate_activation is not None:
                # print(gate_activation.shape, len(gate_activations))
        return input, gate_activations

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, patch_groups = 1):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.do_patch = stride==1 and inplanes<=1024
        if not self.do_patch:
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.bn1 = norm_layer(planes)
            self.conv2 = conv3x3(planes, planes)
        else:
            self.patch_groups = patch_groups
            self.conv3x3_gs = nn.Sequential(
                nn.Conv2d(inplanes, patch_groups*4,kernel_size=3, padding=1, stride=1, bias=False, groups = patch_groups),
                nn.BatchNorm2d(patch_groups*4),
                nn.ReLU(inplace=True)
            )
            self.pool7x7 = nn.AdaptiveAvgPool2d((7,7))
            self.fc_gs = nn.Conv2d(patch_groups*4,patch_groups*2,kernel_size=1,stride=1,padding=0,bias=True, groups = patch_groups)
            self.fc_gs.bias.data[:patch_groups] = 1.0
            self.fc_gs.bias.data[patch_groups:] = 10.0            
            self.gs = GumbleSoftmax()

            self.downsample_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            self.conv1_base = conv3x3(inplanes, planes)
            self.bn1_base = norm_layer(planes)
            self.conv2_base = conv3x3(planes, planes)

            self.conv1_refine = conv3x3(inplanes, planes, groups=patch_groups)
            self.bn1_refine = norm_layer(planes)
            self.conv2_refine = conv3x3(planes, planes, groups=patch_groups)

        self.relu = nn.ReLU(inplace=True)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, temperature = 1):
        identity = x
        if not self.do_patch:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
        else:
            b,c,h,w = x.shape
            x_base = self.downsample_pool(x)
            x_base = self.conv1_base(x_base)
            x_base = self.bn1_base(x_base)
            x_base = self.relu(x_base)
            x_base = self.conv2_base(x_base)
            x_base = F.interpolate(x_base, size = (h,w), mode = 'nearest')
            
            gates = self.conv3x3_gs(x)
            gates = self.pool7x7(gates)
            gates = self.fc_gs(gates)
            gates = gates.view(b,2,self.patch_groups,7,7)
            gates = self.gs(gates, temp=temperature, force_hard=True)
            gates = gates[:,1,:,:,:]
            gates_expand = gates
            if self.patch_groups > 1:
                gates_expand = gates.unsqueeze(1).repeat(1,c//self.patch_groups,1,1,1).transpose(1,2).reshape(b,c,7,7)
                
            x_refine = x * F.interpolate(gates_expand, size = (h,w))
            x_refine = self.conv1_refine(x_refine)
            x_refine = self.bn1_refine(x_refine)
            x_refine = self.relu(x_refine)
            x_refine = self.conv2_refine(x_refine)

            out = x_base + x_refine
        
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        if self.do_patch:
            return out, gates
        else:
            return out, None

    def forward_calc_flops(self, x, temperature = 1):
        flops = 0
        identity = x
        if not self.do_patch:
            c_in = x.shape[1]
            out = self.conv1(x)
            flops += c_in * out.shape[1] * out.shape[2] * out.shape[3] * 9
            out = self.bn1(out)
            out = self.relu(out)
            flops += out[0].numel()
            c_in = out.shape[1]
            out = self.conv2(out)
            flops += c_in * out.shape[1] * out.shape[2] * out.shape[3] * 9
        else:
            b,c,h,w = x.shape
            x_base = self.downsample_pool(x)
            flops += c * x_base.shape[2] * x_base.shape[3] * 9
            c_in = x_base.shape[1]
            x_base = self.conv1_base(x_base)
            flops += c_in * x_base.shape[1] * x_base.shape[2] * x_base.shape[3] * 9
            x_base = self.bn1_base(x_base)
            x_base = self.relu(x_base)
            flops += x_base[0].numel()
            c_in = x_base.shape[1]
            x_base = self.conv2_base(x_base)
            flops += c_in * x_base.shape[1] * x_base.shape[2] * x_base.shape[3] * 9
            x_base = F.interpolate(x_base, size = (h,w), mode = 'nearest')
            
            c_in = x.shape[1]
            gates = self.conv3x3_gs(x)
            flops += c_in * gates.shape[1] * gates.shape[2] * gates.shape[3] * 9 / self.patch_groups

            flops += gates.shape[1] * gates.shape[2] * gates.shape[3]
            gates = self.pool7x7(gates)

            c_in = gates.shape[1]
            gates = self.fc_gs(gates)
            flops += c_in * gates.shape[1] * gates.shape[2] * gates.shape[3] / self.patch_groups

            gates = gates.view(b,2,self.patch_groups,7,7)
            gates = self.gs(gates, temp=temperature, force_hard=True)
            gates = gates[:,1,:,:,:]
            
            gates_expand = gates
            if self.patch_groups > 1:
                gates_expand = gates.unsqueeze(1).repeat(1,c//self.patch_groups,1,1,1).transpose(1,2).reshape(b,c,7,7)
                
            x_refine = x * F.interpolate(gates_expand, size = (h,w))
            c_in = x_refine.shape[1]
            ratio = torch.mean(gates)
            # ratio = 0.729
            x_refine = self.conv1_refine(x_refine)
            flops += ratio * c_in * x_refine.shape[1] * x_refine.shape[2] * x_refine.shape[3] * 9 / self.patch_groups
            x_refine = self.bn1_refine(x_refine)
            x_refine = self.relu(x_refine)
            flops += ratio * x_refine[0].numel()
            c_in = x_refine.shape[1]
            x_refine = self.conv2_refine(x_refine)
            flops += ratio * c_in * x_refine.shape[1] * x_refine.shape[2] * x_refine.shape[3] * 9 / self.patch_groups
            out = x_base + x_refine
        
        out = self.bn2(out)
        if self.downsample is not None:
            c_in = x.shape[1]
            identity = self.downsample(x)
            flops += c_in * identity.shape[1] * identity.shape[2] * identity.shape[3]
        out += identity
        out = self.relu(out)
        flops += out[0].numel()
        if self.do_patch:
            return out, gates, flops
        else:
            return out, None, flops

class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, patch_groups = 1):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.do_patch = stride==1 and inplanes<=1024
        print(self.do_patch)
        if self.do_patch:
            self.patch_groups = patch_groups
            self.conv3x3_gs = nn.Sequential(
                nn.Conv2d(inplanes, patch_groups*4,kernel_size=3, padding=1, stride=1, bias=False, groups = patch_groups),
                nn.BatchNorm2d(patch_groups*4),
                nn.ReLU(inplace=True)
            )
            self.pool7x7 = nn.AdaptiveAvgPool2d((7,7))
            self.fc_gs = nn.Conv2d(patch_groups*4,patch_groups*2,kernel_size=1,stride=1,padding=0,bias=True, groups = patch_groups)
            self.fc_gs.bias.data[:patch_groups] = 1.0
            self.fc_gs.bias.data[patch_groups:] = 10.0            
            self.gs = GumbleSoftmax()

            self.downsample_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            self.conv1_base = conv1x1(inplanes, width)
            self.bn1_base = norm_layer(width)
            self.conv2_base = conv3x3(width, width, stride, groups, dilation)
            self.bn2_base = norm_layer(width)
            self.conv3_base = conv1x1(width, planes * self.expansion)

            self.conv1_refine = conv1x1(inplanes, width, groups=patch_groups)
            self.bn1_refine = norm_layer(width)
            self.conv2_refine = conv3x3(width, width, stride,groups=patch_groups, dilation=dilation)
            self.bn2_refine = norm_layer(width)
            self.conv3_refine = conv1x1(width, planes * self.expansion, groups=patch_groups)

        else:
            self.conv1 = conv1x1(inplanes, width)
            self.bn1 = norm_layer(width)
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
            self.bn2 = norm_layer(width)
            self.conv3 = conv1x1(width, planes * self.expansion)

        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, temperature = 1.0):
        identity = x
        # print(x.shape)
        if self.do_patch:
            b,c,h,w = x.shape
            x_base = self.downsample_pool(x)
            x_base = self.relu(self.bn1_base(self.conv1_base(x_base)))
            x_base = self.relu(self.bn2_base(self.conv2_base(x_base)))
            x_base = self.conv3_base(x_base)
            x_base = F.interpolate(x_base, size = (h,w), mode = 'nearest')

            gates = self.conv3x3_gs(x)
            gates = self.pool7x7(gates)
            gates = self.fc_gs(gates)
            gates = gates.view(b,2,self.patch_groups,7,7)
            gates = self.gs(gates, temp=temperature, force_hard=True)
            gates = gates[:,1,:,:,:]
            gates_expand = gates
            if self.patch_groups > 1:
                gates_expand = gates.unsqueeze(1).repeat(1,c//self.patch_groups,1,1,1).transpose(1,2).reshape(b,c,7,7)
                
            x_refine = x * F.interpolate(gates_expand, size = (h,w))
            x_refine = self.relu(self.bn1_refine(self.conv1_refine(x_refine)))
            x_refine = self.relu(self.bn2_refine(self.conv2_refine(x_refine)))
            x_refine = self.conv3_refine(x_refine)
            x_base += x_refine
            
        else:
            x_base = self.relu(self.bn1(self.conv1(x)))
            x_base = self.conv2(x_base)
            x_base = self.relu(self.bn2(x_base))
            x_base = self.conv3(x_base)
        
        x_base = self.bn3(x_base)
        if self.downsample is not None:
            identity = self.downsample(x)

        x_base += identity
        x_base = self.relu(x_base)
        if self.do_patch:
            return x_base, gates
        else:
            return x_base, None

    def forward_calc_flops(self, x, temperature = 1):
        identity = x
        # flops = torch.tensor(0.0).cuda().detach()
        flops = 0
        # print(self.do_patch)
        if self.do_patch:
            b,c,h,w = x.shape
            # x_base = x
            x_base = self.downsample_pool(x)
            flops += c * x_base.shape[2] * x_base.shape[3] * 9

            x_base = self.relu(self.bn1_base(self.conv1_base(x_base)))
            flops += c * x_base.shape[1] * x_base.shape[2] * x_base.shape[3]
            flops += x_base.numel() / b

            c_in = x_base.shape[1]
            x_base = self.relu(self.bn2_base(self.conv2_base(x_base)))
            flops += c_in * x_base.shape[1] * x_base.shape[2] * x_base.shape[3] * 9
            
            c_in = x_base.shape[1]
            x_base = self.conv3_base(x_base)
            flops += c_in * x_base.shape[1] * x_base.shape[2] * x_base.shape[3]

            x_base = F.interpolate(x_base, size = (h,w), mode = 'nearest')


            gates = self.conv3x3_gs(x)
            flops += c * gates.shape[1] * gates.shape[2] * gates.shape[3] * 9 / self.patch_groups
            
            flops += gates.shape[1] * gates.shape[2] * gates.shape[3]
            gates = self.pool7x7(gates)
            
            # print(gates.shape)
            c_in = gates.shape[1]
            gates = self.fc_gs(gates)
            flops += c_in * gates.shape[1] * gates.shape[2] * gates.shape[3] / self.patch_groups

            # print(gates.shape)
            gates = gates.view(b,2,self.patch_groups,7,7)
            gates = self.gs(gates, temp=temperature, force_hard=True)
            gates = gates[:,1,:,:,:]
            ratio = torch.mean(gates)
            # ratio = 0.7
            
            gates_expand = gates
            # print(gates.shape)
            if self.patch_groups > 1:
                gates_expand = gates.unsqueeze(1).repeat(1,c//self.patch_groups,1,1,1).transpose(1,2).reshape(b,c,7,7)
                
            x_refine = x * F.interpolate(gates_expand, size = (h,w))
            c_in = x_refine.shape[1]
            x_refine = self.relu(self.bn1_refine(self.conv1_refine(x_refine)))
            flops += ratio * c_in * x_refine.shape[1] * x_refine.shape[2] * x_refine.shape[3] / self.patch_groups
            flops += ratio * x_refine[0].numel()

            c_in = x_refine.shape[1]
            x_refine = self.relu(self.bn2_refine(self.conv2_refine(x_refine)))
            flops += ratio * c_in * x_refine.shape[1] * x_refine.shape[2] * x_refine.shape[3] * 9 / self.patch_groups
            flops += ratio * x_refine[0].numel()

            c_in = x_refine.shape[1]
            x_refine = self.conv3_refine(x_refine)
            flops += ratio * c_in * x_refine.shape[1] * x_refine.shape[2] * x_refine.shape[3] / self.patch_groups
            flops += ratio * x_refine[0].numel()

            x_base += x_refine
            
            # assert(0==1)
        else:
            c_in = x.shape[1]
            x_base = self.relu(self.bn1(self.conv1(x)))
            flops += c_in * x_base.shape[1] * x_base.shape[2] * x_base.shape[3]
            flops += x_base[0].numel()

            c_in = x_base.shape[1]
            x_base = self.relu(self.bn2(self.conv2(x_base)))
            flops += c_in * x_base.shape[1] * x_base.shape[2] * x_base.shape[3] * 9
            flops += x_base[0].numel()

            c_in = x_base.shape[1]
            x_base = self.conv3(x_base)
            flops += c_in * x_base.shape[1] * x_base.shape[2] * x_base.shape[3]

        x_base = self.bn3(x_base)

        if self.downsample is not None:
            c_in = x.shape[1]
            identity = self.downsample(x)
            flops += c_in * identity.shape[1] * identity.shape[2] * identity.shape[3]
        x_base += identity
        x_base = self.relu(x_base)
        flops += x_base[0].numel()
        # print(flops/1e8)
        if self.do_patch:
            return x_base, gates, flops
        else:
            return x_base, None, flops

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, patch_groups = 8):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.patch_groups = patch_groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        for k, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if 'gs' in str(k):
                    m.weight.data.normal_(0, 0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck):
        #             nn.init.constant_(m.bn3.weight, 0)
        #         elif isinstance(m, BasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, patch_groups = self.patch_groups))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, patch_groups = self.patch_groups))

        return Sequential_ext(*layers)

    def forward(self, x, temperature=1.0):
        # See note [TorchScript super()]
        gate_activations = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print('layer 1')
        x, a = self.layer1(x, temperature)
        gate_activations.extend(a)
        # print(len(a), len(gate_activations))
        # print('layer 2')
        x, a = self.layer2(x, temperature)
        gate_activations.extend(a)
        # print(len(a), len(gate_activations))
        # print('layer 3')
        x, a = self.layer3(x, temperature)
        gate_activations.extend(a)
        # print('layer 4')
        x, _ = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, gate_activations

    def forward_calc_flops(self, x, temperature=1.0):

        # x1, _ = self.forward(x)

        # flops = torch.tensor(0.0).cuda().detach()
        flops = 0
        gate_activations = []
        b,c_in, _, _ = x.shape
        x = self.conv1(x)
        flops += c_in * x.shape[1] * x.shape[2] * x.shape[3] * 49
        x = self.bn1(x)
        x = self.relu(x)
        flops += x.numel() / b

        x = self.maxpool(x)
        flops += x.shape[1] * x.shape[2] * x.shape[3] * 9

        for i in range(len(self.layer1)):
            x, a, tmp_flops = self.layer1[i].forward_calc_flops(x, temperature)
            gate_activations.append(a)
            flops += tmp_flops
        for i in range(len(self.layer2)):
            x, a, tmp_flops = self.layer2[i].forward_calc_flops(x, temperature)
            if a is not None:
                gate_activations.append(a)
            flops += tmp_flops
        for i in range(len(self.layer3)):
            x, a, tmp_flops = self.layer3[i].forward_calc_flops(x, temperature)
            if a is not None:
                gate_activations.append(a)
            flops += tmp_flops
        # print(flops/1e9)
        for i in range(len(self.layer4)):
            x, a, tmp_flops = self.layer4[i].forward_calc_flops(x, temperature)
            if a is not None:
                gate_activations.append(a)
            flops += tmp_flops

        flops += x.shape[1] * x.shape[2] * x.shape[3]
        x = self.avgpool(x)
        
        x = torch.flatten(x, 1)
        c_in = x.shape[1]
        x = self.fc(x)
        flops += c_in * x.shape[1]

        # x, gate_activations = self.forward(x)
        return x, gate_activations, flops


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model


def resnet18(args):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained = False, progress=True)



def resnet34(args):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained = False, progress=True, patch_groups = args.patch_groups)


def resnet50(args):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained = False, progress=True, patch_groups = args.patch_groups)

def resnet101(args):
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained = False, progress=True, patch_groups = args.patch_groups)


def resnet152(args):
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3],pretrained = False, progress=True, patch_groups = args.patch_groups)


def resnext50_32x4d(args):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], pretrained = False, progress=True)


def resnext101_32x8d(args):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],pretrained = False, progress=True, patch_groups = args.patch_groups)



def wide_resnet50_2(args):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],pretrained = False, progress=True, width_per_group=64*2, patch_groups = args.patch_groups)



def wide_resnet101_2(args):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],pretrained = False, progress=True, patch_groups = args.patch_groups)



if __name__ == "__main__":
    import numpy as np
    def params_count(model):
        return np.sum([p.numel() for p in model.parameters()]).item()
    import argparse
    from op_counter import measure_model
    parser = argparse.ArgumentParser(description='PyTorch resnet Training')
    args = parser.parse_args()

    args.num_classes = 1000
    args.patch_groups = 4
    net = resnet50(args)
    # y, gate_activation = net(torch.rand(6,3,224,224), temperature=0.001)
    # print(len(gate_activation))
    # _,gate_activation,flops = net.forward_calc_flops(torch.rand(1,3,224,224), temperature=0.001)
    # gate_activation = [a for a in gate_activation if a is not None]
    # print(len(gate_activation))
    # act_rate = 0.0
    # for act in gate_activation:
    #     act_rate += torch.mean(act)
    # act_rate = torch.mean(act_rate/len(gate_activation))
    # num_of_params = params_count(net)
    # print(act_rate,num_of_params/1e6, flops / 1e9)