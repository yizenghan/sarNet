from __future__ import division

import argparse
import pprint
from mmcv import Config

from mmcls import __version__
from mmcls.datasets import get_dataset
from mmcls.apis import (train_classifier, init_dist, get_root_logger,
                        set_random_seed)
from mmcls.models import build_classifier
import torch
import torch.distributed as dist
from mmcls.utils import get_git_hash, summary


def parse_args():
    parser = argparse.ArgumentParser(description='Train a classifier')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument('--data_root', help='the data root dir')
    parser.add_argument('--cache_mode', help='whether use cache mode', action='store_true')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.data_root is not None:
        cfg.data_root = args.data_root
        cfg.data.train.data_root = args.data_root
        cfg.data.val.data_root = args.data_root
    if args.cache_mode:
        cfg.data.train.cache_mode = True
        cfg.data.val.cache_mode = True
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus
    if cfg.checkpoint_config is not None:
        # save mmcls version in checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmcls_version=__version__, config=cfg.text)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    # update gpu num
    if dist.is_initialized():
        cfg.gpus = dist.get_world_size()
    cfg.optimizer.lr *= cfg.gpus/4.
    if cfg.lr_config.get('warmup', None) is not None:
        cfg.lr_config.warmup_iters /= cfg.gpus/4.
    # print cfg
    logger.info('training config:{}\n'.format(pprint.pformat(cfg._cfg_dict)))

    # pring git hash
    logger.info('git_log: {}'.format(get_git_hash()))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    model = build_classifier(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    try:
        logger.info(summary(model, cfg))
    except RuntimeError:
        logger.info('RuntimeError during summary')
        logger.info(str(model))

    train_dataset = get_dataset(cfg.data.train)
    if args.validate and ('val', 1) in cfg.workflow:
        val_dataset = get_dataset(cfg.data.val)
        if ('val', 1) not in cfg.workflow:
            cfg.work_flow.append(('val', 1))
    else:
        val_dataset = None
        if ('val', 1) in cfg.workflow:
            cfg.workflow.remove(('val', 1))
    train_classifier(
        model,
        train_dataset,
        val_dataset,
        cfg,
        distributed=distributed,
        logger=logger)


if __name__ == '__main__':
    main()
