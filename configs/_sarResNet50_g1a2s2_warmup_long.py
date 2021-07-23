model = dict(
    type='models.sar_resnet_alpha',
    num_classes=1000,
    depth=50,
    mask_size=7,
    patch_groups=1,
    alpha=2,
    base_scale=2
)
train_cfg = dict(
    hyperparams_set_index=235,
    crop_size=224,
)
test_cfg = dict(
    crop_size=224,
    crop_type='normal'
)