model = dict(
    type='models.sar_resnet_freFuse',
    num_classes=1000,
    depth=50,
    patch_groups=2,
    alpha=1,
    base_scale=2,
    mask_size1=7,
    mask_size2=1    
)
train_cfg = dict(
    hyperparams_set_index=233,
    crop_size=224,
)
test_cfg = dict(
    crop_size=224,
    crop_type='normal'
)