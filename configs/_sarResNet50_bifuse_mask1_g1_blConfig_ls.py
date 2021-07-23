model = dict(
    type='models.sar_resnet_mask1_bifuse',
    num_classes=1000,
    depth=50,
    patch_groups=1,
    width=1, 
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