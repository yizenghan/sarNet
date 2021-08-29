from __future__ import division

from collections import OrderedDict

import torch
from mmcv.runner import DistSamplerSeedHook
from mmcv_custom import Runner
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmcls.core import DistOptimizerHook
from mmcls.datasets import build_dataloader
from .env import get_root_logger


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for name in log_vars:
        log_vars[name] = log_vars[name].item()

    return loss, log_vars


def batch_processor(model, data, train_mode):
    losses = model(**data)
    loss, log_vars = parse_losses(losses)

    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

    return outputs


def train_classifier(model,
                     train_dataset,
                     val_dataset,
                     cfg,
                     distributed=False,
                     validate=False,
                     logger=None):
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # start training
    if distributed:
        _dist_train(model, train_dataset, val_dataset, cfg)
    else:
        _non_dist_train(model, train_dataset, val_dataset, cfg)


def _dist_train(model, train_dataset, val_dataset, cfg):
    # prepare data loaders
    data_loaders = [
        build_dataloader(
            train_dataset,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            dist=True)
    ]
    if val_dataset is not None:
        data_loaders.append(
            build_dataloader(
                val_dataset,
                cfg.data.imgs_per_gpu,
                cfg.data.workers_per_gpu,
                dist=True)
        )
    # put model on gpus
    model = MMDistributedDataParallel(model.cuda())
    # build runner
    runner = Runner(model, batch_processor, cfg.optimizer, cfg.work_dir,
                    cfg.log_level)
    # register hooks
    optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)
    runner.register_hook(DistSamplerSeedHook())

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.auto_resume:
        runner.auto_resume()
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


def _non_dist_train(model, train_dataset, val_dataset, cfg):
    # prepare data loaders
    data_loaders = [
        build_dataloader(
            train_dataset,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            cfg.gpus,
            dist=False)
    ]
    if val_dataset is not None:
        data_loaders.append(
            build_dataloader(
                val_dataset,
                cfg.data.imgs_per_gpu,
                cfg.data.workers_per_gpu,
                dist=False)
        )
    # put model on gpus
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()
    # build runner
    runner = Runner(model, batch_processor, cfg.optimizer, cfg.work_dir,
                    cfg.log_level)
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.auto_resume:
        runner.auto_resume()
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
