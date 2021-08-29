import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcls.core import weighted_cross_entropy, accuracy
from ..registry import HEADS


@HEADS.register_module
class FCHead(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively"""

    def __init__(self,
                 with_avg_pool=True,
                 in_channels=2048,
                 roi_feat_size=None,
                 dropout=None,
                 dim=2,
                 num_classes=10):
        super(FCHead, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.in_channels = in_channels
        self.roi_feat_size = roi_feat_size
        self.num_classes = num_classes
        assert dim in [2, 3], 'dim {} must in [2, 3]'.format(dim)

        in_channels = self.in_channels
        if self.with_avg_pool:
            if dim == 2:
                if roi_feat_size is None:
                    self.avg_pool = nn.AdaptiveAvgPool2d(1)
                else:
                    self.avg_pool = nn.AvgPool2d(roi_feat_size)
            else:
                if roi_feat_size is None:
                    self.avg_pool = nn.AdaptiveAvgPool3d(1)
                else:
                    self.avg_pool = nn.AvgPool3d(roi_feat_size)
        else:
            in_channels *= (self.roi_feat_size * self.roi_feat_size)
        self.fc_cls = nn.Linear(in_channels, num_classes)
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

    def init_weights(self):
        nn.init.normal_(self.fc_cls.weight, 0, 0.01)
        nn.init.constant_(self.fc_cls.bias, 0)

    def forward(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        if self.dropout is not None:
            x = self.dropout(x)
        cls_score = self.fc_cls(x)
        return cls_score

    def loss(self,
             cls_score,
             labels,
             reduce=True):
        losses = dict()
        label_weights = cls_score.new_ones(labels.size())
        losses['loss_cls'] = weighted_cross_entropy(
            cls_score, labels, label_weights, reduce=reduce)
        losses['acc_top1'] = accuracy(cls_score, labels, topk=1)
        losses['acc_top5'] = accuracy(cls_score, labels, topk=5)
        return losses

