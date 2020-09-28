model = dict(
    type='double_checked_models.resnest50_conv7x7',
    pretrained=False,
    dropout=0.0,
    dropblock_prob=0.0
)
train_cfg = dict(
    hyperparams_set_index=1,
    crop_size=224,
)
test_cfg = dict(
    crop_size=224,
    crop_type='normal'
)