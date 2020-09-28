import double_checked_models.RegNet.core.config as config
from .core.builders import build_model
from .core.config import cfg

import os
from os import path
abspath = path.dirname(__file__)

from torch.hub import load_state_dict_from_url

__all__ = ['effnet_b0', 'effnet_b3', 'effnet_b4', 'effnet_b5']

model_urls = {
    'effnet_b0': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/161305613/EN-B0_dds_8gpu.pyth',
    'effnet_b3': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/161305060/EN-B3_dds_8gpu.pyth',
    'effnet_b4': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/161305098/EN-B4_dds_8gpu.pyth',
    'effnet_b5': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/161305138/EN-B5_dds_8gpu.pyth',
}


def effnet_b0(se=0.0, dropout=0.0, drop_connect_rate=0.0, pretrained=False):
    model_cfg_dir = os.path.join(abspath, 'configs/effnet')
    config.load_cfg(model_cfg_dir, 'EN-B0_dds_8gpu.yaml')
    cfg.EN.SE_R = se
    cfg.EN.DC_RATIO = dropout
    cfg.EN.DROPOUT_RATIO = drop_connect_rate
    cfg.freeze()
    model = build_model()
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['effnet_b0'], progress=True)
        model.load_state_dict(state_dict['model_state'])
    return model


def effnet_b3(se=0.0, dropout=0.0, drop_connect_rate=0.0, pretrained=False):
    model_cfg_dir = os.path.join(abspath, 'configs/effnet')
    config.load_cfg(model_cfg_dir, 'EN-B3_dds_8gpu.yaml')
    cfg.EN.SE_R = se
    cfg.EN.DC_RATIO = dropout
    cfg.EN.DROPOUT_RATIO = drop_connect_rate
    cfg.freeze()
    model = build_model()
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['effnet_b3'], progress=True)
        model.load_state_dict(state_dict['model_state'])
    return model


def effnet_b4(se=0.0, dropout=0.0, drop_connect_rate=0.0, pretrained=False):
    model_cfg_dir = os.path.join(abspath, 'configs/effnet')
    config.load_cfg(model_cfg_dir, 'EN-B4_dds_8gpu.yaml')
    cfg.EN.SE_R = se
    cfg.EN.DC_RATIO = dropout
    cfg.EN.DROPOUT_RATIO = drop_connect_rate
    cfg.freeze()
    model = build_model()
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['effnet_b4'], progress=True)
        model.load_state_dict(state_dict['model_state'])
    return model


def effnet_b5(se=0.0, dropout=0.0, drop_connect_rate=0.0, pretrained=False):
    model_cfg_dir = os.path.join(abspath, 'configs/effnet')
    config.load_cfg(model_cfg_dir, 'EN-B5_dds_8gpu.yaml')
    cfg.EN.SE_R = se
    cfg.EN.DC_RATIO = dropout
    cfg.EN.DROPOUT_RATIO = drop_connect_rate
    cfg.freeze()
    model = build_model()
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['effnet_b5'], progress=True)
        model.load_state_dict(state_dict['model_state'])
    return model