model = dict(
    type='models.sar_resnet',
    num_classes=1000,
    depth=50,
    mask_size=7,
    patch_groups=4,
    alpha=2
)
train_cfg = dict(
    hyperparams_set_index=233,
    crop_size=224,
)
test_cfg = dict(
    crop_size=224,
    crop_type='normal'
)