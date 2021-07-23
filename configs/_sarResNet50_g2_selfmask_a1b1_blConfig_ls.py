model = dict(
    type='models.sar_resnet_alpha_selfmask',
    num_classes=1000,
    depth=50,
    mask_size=7,
    patch_groups=2,
    alpha=1
)
train_cfg = dict(
    hyperparams_set_index=333,
    crop_size=224,
)
test_cfg = dict(
    crop_size=224,
    crop_type='normal'
)