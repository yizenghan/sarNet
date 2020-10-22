model = dict(
    type='models.resnet34',
)
train_cfg = dict(
    hyperparams_set_index=2333,
    crop_size=224,
)
test_cfg = dict(
    crop_size=224,
    crop_type='normal'
)