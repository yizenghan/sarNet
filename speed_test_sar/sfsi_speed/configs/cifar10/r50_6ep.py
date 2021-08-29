# model settings
model = dict(
    type='VanillaNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
    ),
    cls_head=dict(
        type='FCHead',
        num_classes=10))

train_cfg = dict(
)
test_cfg = dict(
)
# dataset settings
dataset_type = 'Cifar10Dataset'
data_root = 'data/cifar10/'
img_norm_cfg = dict(
    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
data = dict(
    imgs_per_gpu=64,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root='data/cifar10/',
        ann_file=None,
        img_prefix=None,
        img_norm_cfg=img_norm_cfg,
        train=True),
    val=dict(
        type=dataset_type,
        data_root='data/cifar10/',
        ann_file=None,
        img_prefix=None,
        img_norm_cfg=img_norm_cfg,
        train=False),
)
# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=1.0 / 3,
#     step=[8, 11])
lr_config = dict(policy='step', step=2)
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 6
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/r50_6ep'
load_from = None
resume_from = None
auto_resume = False
workflow = [('train', 1), ('val', 1)]
# dataset settings
# data_root = '/mnt/SSD/dataset/cifar10'
# mean = [0.4914, 0.4822, 0.4465]
# std = [0.2023, 0.1994, 0.2010]

