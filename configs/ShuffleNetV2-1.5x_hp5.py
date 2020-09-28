model = dict(
    type='double_checked_models.shufflenetv2_15x',
    num_classes=1000,
    dropout=0.0,
    pretrained=False,
)
train_cfg = dict(
    hyperparams_set_index=5,
    crop_size=224,
)
test_cfg = dict(
    crop_size=224,
    crop_type='normal'
)