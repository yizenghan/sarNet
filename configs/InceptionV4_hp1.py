model = dict(
    type='double_checked_models.inceptionv4',
    num_classes=1000,
    pretrained='imagenet',
    dropout=0.0
)
train_cfg = dict(
    hyperparams_set_index=1,
    crop_size=299,
)
test_cfg = dict(
    crop_size=299,
    crop_type='resnest'
)