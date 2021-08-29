import torch
from torch import nn
from . import select_feature_cython
# from mmdet.utils import GlobalDict

import numpy as np
GlobalDict = dict()
def SelectFeature(input, offset):
    output = np.zeros((input.shape[0], input.shape[1], offset.shape[1], offset.shape[2]), dtype=np.float32)
    # select_feature_cython.apply_select_features(input.float().numpy(), offset.int().numpy(), GlobalDict.GlobalDict['thread_num'], output)
    # print(input.device)
    if 'cuda' in str(input.device):
        select_feature_cython.apply_select_features(input.float().cpu().numpy(), offset.int().cpu().numpy(), torch.get_num_threads(), output)
        output = torch.Tensor(output)
        output = output.cuda(device=input.device)
    else:
        select_feature_cython.apply_select_features(input.float().numpy(), offset.int().numpy(), torch.get_num_threads(), output)
        output = torch.Tensor(output)
    return output

def SelectFeatureBiliner(input, offset):
    pass
