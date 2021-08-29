import logging
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from mmcv.runner import get_dist_info
from mmcls.utils import (gpu_indices, ompi_size, ompi_rank, get_master_ip,
                         ompi_local_size, ompi_local_rank, get_local_rank, get_local_size)

def init_dist(launcher, backend='nccl', **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'mpi':
        _init_dist_mpi(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError('Invalid launcher type: {}'.format(launcher))


def _init_dist_pytorch(backend, **kwargs):
    # TODO: use local_rank instead of rank % num_gpus
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    local_rank = rank % num_gpus
    os.environ['LOCAL_RANK'] = str(local_rank)
    local_size = num_gpus
    os.environ['LOCAL_SIZE'] = str(local_size)
    print('rank:{} local_size:{} local_rank:{}'.format(rank, local_size, local_rank))
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def _init_dist_mpi(backend, **kwargs):
    gpus = list(gpu_indices())
    gpu_num = len(gpus)
    world_size = ompi_size()
    rank = ompi_rank()
    os.environ['LOCAL_SIZE'] = str(ompi_local_size())
    os.environ['LOCAL_RANK'] = str(ompi_local_rank())
    dist_url = 'tcp://' + get_master_ip() + ':23456'
    torch.cuda.set_device(int(gpus[0]))  # Set current GPU to the first
    dist.init_process_group(
        backend=backend,
        init_method=dist_url,
        world_size=world_size,
        rank=rank,
        group_name='mtorch'
    )
    print('rank: {} local_size: {} local_rank: {}'.format(rank, get_local_size(), get_local_rank()))
    print("World Size is {}, Backend is {}, Init Method is {}, gpu num is{}".format(world_size, backend,
                                                                                    dist_url, gpu_num))


def _init_dist_slurm(backend, **kwargs):
    raise NotImplementedError


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_root_logger(log_level=logging.INFO):
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=log_level)
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel('ERROR')
    return logger
