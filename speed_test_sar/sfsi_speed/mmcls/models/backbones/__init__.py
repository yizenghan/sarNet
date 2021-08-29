#from .resnet import ResNet
#from .resnext import ResNeXt
from .scalenet import ScaleNet
from .resnet_downsample import ResNetDownSample
from .lcclnet import LCCLNet
from .scalenet_wointerpolation import ScaleNetWoInterpolation

__all__ = ['ScaleNet', 'LCCLNet', 
           'ScaleNetWoInterpolation', 'ResNetDownSample', 'SelectFeature']
