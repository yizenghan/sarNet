from mmcv.runner import get_dist_info

def is_host():
    return get_dist_info()[0] == 0

Sparsity = dict()