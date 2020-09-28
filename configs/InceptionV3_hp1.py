model = dict(
    type='double_checked_models.inception_v3',
    num_classes=1000,
    pretrained=False,
    dropout=0.0,
    aux_logits=True,
    transform_input=False
)
train_cfg = dict(
    hyperparams_set_index=1,
    crop_size=299,
)
test_cfg = dict(
    crop_size=299,
    crop_type='normal'
)