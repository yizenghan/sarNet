model = dict(
    type='models.oct_resnet50',
)
train_cfg = dict(
    hyperparams_set_index=234,
    crop_size=224,
)
test_cfg = dict(
    crop_size=224,
    crop_type='normal'
)