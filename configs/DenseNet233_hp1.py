model = dict(
    type='double_checked_models.densenet233',
    num_classes=1000,
    pretrained=False,
    progress=True,
    memory_efficient=False,
    dropout=0.0
)
train_cfg = dict(
    hyperparams_set_index=1,
    crop_size=224,
)
test_cfg = dict(
    crop_size=224,
    crop_type='normal'
)