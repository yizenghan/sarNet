import os.path as osp
import mmcv
from mmcv.runner.utils import obj_from_dict
from mmcv.runner.hooks import (Hook, LrUpdaterHook, CheckpointHook, IterTimerHook,
                    OptimizerHook, lr_updater)
from mmcv_custom import lr_updater as lr_updater_custom
from mmcv.runner.checkpoint import save_checkpoint

import torch
from .parameters import parameters


class Runner(mmcv.runner.Runner):
    """A training helper for PyTorch.

        Custom version of mmcv runner, overwrite init_optimizer method
    """
    def init_optimizer(self, optimizer):
        """Init the optimizer.

        Args:
            optimizer (dict or :obj:`~torch.optim.Optimizer`): Either an
                optimizer object or a dict used for constructing the optimizer.

        Returns:
            :obj:`~torch.optim.Optimizer`: An optimizer object.

        Examples:
            >>> optimizer = dict(type='SGD', lr=0.01, momentum=0.9)
            >>> type(runner.init_optimizer(optimizer))
            <class 'torch.optim.sgd.SGD'>
        """
        if isinstance(optimizer, dict):
            optimizer = obj_from_dict(
                optimizer, torch.optim, dict(params=parameters(self.model, optimizer.lr)))
        elif not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                'optimizer must be either an Optimizer object or a dict, '
                'but got {}'.format(type(optimizer)))
        return optimizer

    def resume(self, checkpoint, resume_optimizer=True,
               map_location='default'):
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = self.load_checkpoint(
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']
        if 'optimizer' in checkpoint and resume_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info('resumed epoch %d, iter %d', self.epoch, self.iter)

    def auto_resume(self):
        linkname = osp.join(self.work_dir, 'latest.pth')
        if osp.exists(linkname):
            self.logger.info('latest checkpoint found')
            self.resume(linkname)

    def register_lr_hooks(self, lr_config):
        if isinstance(lr_config, LrUpdaterHook):
            self.register_hook(lr_config)
        elif isinstance(lr_config, dict):
            assert 'policy' in lr_config
            # from .hooks import lr_updater
            hook_name = lr_config['policy'].title() + 'LrUpdaterHook'
            if hasattr(lr_updater, hook_name):
                hook_cls = getattr(lr_updater, hook_name)
            elif hasattr(lr_updater_custom, hook_name):
                hook_cls = getattr(lr_updater_custom, hook_name)
            else:
                raise ValueError('"{}" does not exist'.format(hook_name))

            self.register_hook(hook_cls(**lr_config))
        else:
            raise TypeError('"lr_config" must be either a LrUpdaterHook object'
                            ' or dict, not {}'.format(type(lr_config)))

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None):
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        else:
            meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
