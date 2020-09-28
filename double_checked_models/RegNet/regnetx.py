import double_checked_models.RegNet.core.config as config
from .core.builders import build_model
from .core.config import cfg

import os
from os import path
abspath = path.dirname(__file__)

from torch.hub import load_state_dict_from_url

__all__ = ['regnetx_02', 'regnetx_04', 'regnetx_06', 'regnetx_16', 'regnetx_40', 'regnetx_80', 'regnetx_120']

model_urls = {
    'regnetx_02': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/160905981/RegNetX-200MF_dds_8gpu.pyth',
    'regnetx_04': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/160905967/RegNetX-400MF_dds_8gpu.pyth',
    'regnetx_06': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906442/RegNetX-600MF_dds_8gpu.pyth',
    'regnetx_16': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/160990626/RegNetX-1.6GF_dds_8gpu.pyth',
    'regnetx_40': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906383/RegNetX-4.0GF_dds_8gpu.pyth',
    'regnetx_80': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/161107726/RegNetX-8.0GF_dds_8gpu.pyth',
    'regnetx_120': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906020/RegNetX-12GF_dds_8gpu.pyth',
}


def regnetx_02(pretrained=False):
    model_cfg_dir = os.path.join(abspath, 'configs/regnetx')
    config.load_cfg(model_cfg_dir, 'RegNetX-200MF_dds_8gpu.yaml')
    cfg.freeze()
    model = build_model()
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['regnetx_02'],
                                              progress=True)
        model.load_state_dict(state_dict['model_state'])
    return model


def regnetx_04(pretrained=False):
    model_cfg_dir = os.path.join(abspath, 'configs/regnetx')
    config.load_cfg(model_cfg_dir, 'RegNetX-400MF_dds_8gpu.yaml')
    cfg.freeze()
    model = build_model()
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['regnetx_04'],
                                              progress=True)
        model.load_state_dict(state_dict['model_state'])
    return model


def regnetx_06(pretrained=False):
    model_cfg_dir = os.path.join(abspath, 'configs/regnetx')
    config.load_cfg(model_cfg_dir, 'RegNetX-600MF_dds_8gpu.yaml')
    cfg.freeze()
    model = build_model()
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['regnetx_06'],
                                              progress=True)
        model.load_state_dict(state_dict['model_state'])
    return model


def regnetx_16(pretrained=False):
    model_cfg_dir = os.path.join(abspath, 'configs/regnetx')
    config.load_cfg(model_cfg_dir, 'RegNetX-1.6GF_dds_8gpu.yaml')
    cfg.freeze()
    model = build_model()
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['regnetx_16'],
                                              progress=True)
        # state_dict = torch.load('/home/jhj/.cache/torch/checkpoints/RegNetX-1.6GF_dds_8gpu.pyth',
        #                         map_location="cpu")
        model.load_state_dict(state_dict['model_state'])
    return model


def regnetx_40(pretrained=False):
    model_cfg_dir = os.path.join(abspath, 'configs/regnetx')
    config.load_cfg(model_cfg_dir, 'RegNetX-4.0GF_dds_8gpu.yaml')
    config.assert_and_infer_cfg()
    cfg.freeze()
    model = build_model()
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['regnetx_40'],
                                              progress=True)
        model.load_state_dict(state_dict['model_state'])
    return model


def regnetx_80(pretrained=False):
    model_cfg_dir = os.path.join(abspath, 'configs/regnetx')
    config.load_cfg(model_cfg_dir, 'RegNetX-8.0GF_dds_8gpu.yaml')
    model = build_model()
    cfg.freeze()
    if pretrained:
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls['regnetx_80'],
                                                  progress=True)
            model.load_state_dict(state_dict['model_state'])
    return model


def regnetx_120(pretrained=False):
    model_cfg_dir = os.path.join(abspath, 'configs/regnetx')
    config.load_cfg(model_cfg_dir, 'RegNetX-12GF_dds_8gpu.yaml')
    model = build_model()
    cfg.freeze()
    if pretrained:
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls['regnetx_120'],
                                                  progress=True)
            model.load_state_dict(state_dict['model_state'])
    return model