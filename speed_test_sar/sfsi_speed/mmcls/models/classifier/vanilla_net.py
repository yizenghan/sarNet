from .single_stage import SingleStageClassifier
from ..registry import CLASSIFIERS


@CLASSIFIERS.register_module
class VanillaNet(SingleStageClassifier):

    def __init__(self,
                 backbone,
                 cls_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(VanillaNet, self).__init__(backbone, cls_head, train_cfg,
                                        test_cfg, pretrained)
