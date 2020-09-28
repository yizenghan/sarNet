model = dict(
    type='double_checked_models.effnet_b5',
    pretrained=False,
    se=0.0,
    dropout=0.0,
    drop_connect_rate=0.0 # Stochastic Depth
)
train_cfg = dict(
    hyperparams_set_index=11,
    crop_size=456,
)
test_cfg = dict(
    crop_size=456,
    crop_type='normal'
)