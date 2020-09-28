model = dict(
    type='double_checked_models.hrnet_w32',
    num_classes=1000,
    # pretrained='/home/jhj/hrnetv2_w32_imagenet_pretrained.pth',
    pretrained=False,
    dropout=0.0
)
train_cfg = dict(
    hyperparams_set_index=1,
    crop_size=224,
)
test_cfg = dict(
    crop_size=224,
    crop_type='normal'
)