model = dict(
    type='double_checked_models.mobilenetv2_14',
    num_classes=1000,
    dropout=0.0,
    pretrained=False,
)
train_cfg = dict(
    hyperparams_set_index=41,
    crop_size=224,
)
test_cfg = dict(
    crop_size=224,
    crop_type='normal'
)