import os.path as osp
import pkgutil
import time
from collections import OrderedDict
from importlib import import_module

import mmcv
import torch
from torch.utils import model_zoo
from mmcv.runner.checkpoint import open_mmlab_model_urls, load_state_dict


def load_checkpoint(model,
                    filename,
                    inflate_func=None,
                    map_location=None,
                    strict=False,
                    logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Either a filepath or URL or modelzoo://xxxxxxx.
        map_location (str): Same as :func:`torch.load`.
        inflate_func (func): inflate function applied to state_dict
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    # load checkpoint from modelzoo or file or url
    if filename.startswith('modelzoo://'):
        import torchvision
        model_urls = dict()
        for _, name, ispkg in pkgutil.walk_packages(
                torchvision.models.__path__):
            if not ispkg:
                _zoo = import_module('torchvision.models.{}'.format(name))
                _urls = getattr(_zoo, 'model_urls')
                model_urls.update(_urls)
        model_name = filename[11:]
        checkpoint = model_zoo.load_url(model_urls[model_name])
    elif filename.startswith('open-mmlab://'):
        model_name = filename[13:]
        checkpoint = model_zoo.load_url(open_mmlab_model_urls[model_name])
    elif filename.startswith(('http://', 'https://')):
        checkpoint = model_zoo.load_url(filename)
    else:
        if not osp.isfile(filename):
            raise IOError('{} is not a checkpoint file'.format(filename))
        checkpoint = torch.load(filename, map_location=map_location)
    # get state_dict from checkpoint
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError(
            'No state_dict found in checkpoint file {}'.format(filename))
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
    if inflate_func is not None:
        state_dict = inflate_func(state_dict)
    # load state_dict
    if hasattr(model, 'module'):
        load_state_dict(model.module, state_dict, strict, logger)
    else:
        load_state_dict(model, state_dict, strict, logger)
    return checkpoint


