model = dict(
    type='double_checked_models.effnet_b4',
    pretrained=False,
    se=0.0,
    dropout=0.0,
    drop_connect_rate=0.0 # Stochastic Depth
)
train_cfg = dict(
    hyperparams_set_index=1,
    crop_size=380,
)
test_cfg = dict(
    crop_size=380,
    crop_type='normal'
)