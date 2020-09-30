model = dict(
    type='double_checked_models.regnetx_16',
    pretrained=False,
)
train_cfg = dict(
    hyperparams_set_index=2,
    crop_size=224,
)
test_cfg = dict(
    crop_size=224,
    crop_type='normal'
)