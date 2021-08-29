import torch.nn as nn

from .base import BaseClassifier
from .. import builder
from ..registry import CLASSIFIERS


@CLASSIFIERS.register_module
class SingleStageClassifier(BaseClassifier):

    def __init__(self,
                 backbone,
                 cls_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageClassifier, self).__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg is None and hasattr(self.test_cfg, 'develop'):
            assert isinstance(self.test_cfg, dict)
            backbone.develop.gumbel = self.test_cfg.develop.gumbel

        num_layers = backbone.depth
        if num_layers in [50, 101]:
            cls_head['in_channels'] = 2048
        else:
            cls_head['in_channels'] = 512

        self.backbone = builder.build_backbone(backbone)
        self.cls_head = builder.build_head(cls_head)


        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(SingleStageClassifier, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.cls_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        return x

    def forward_train(self, img, gt_label):
        x = self.extract_feat(img)
        losses = dict()
        if len(x) == 2:
            x, loss_backbone = x
            losses.update({'loss_backbone': loss_backbone})

        outs = self.cls_head(x)
        loss_inputs = (outs, gt_label, )
        losses_head = self.cls_head.loss(*loss_inputs)
        losses.update(losses_head)
        return losses

    def simple_test(self, img):
        x = self.extract_feat(img)
        if len(x) == 2:
            x, _ = x
        outs = self.cls_head(x)
        return outs

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
