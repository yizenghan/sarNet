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
        num_classes=1000))

train_cfg = dict(
)
test_cfg = dict(
)
# dataset settings
dataset_type = 'ImageNetDataset'
#data_root = '/data/home/zhez/mnt/VC_IMAGE_RECOGNITION/Users/v-zhxia/Data/ImageNet-Zip/'
data_root = '/hdfs/public/imagenet/2012/'
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
data = dict(
    imgs_per_gpu=64,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train_map.txt',
        img_prefix='train.zip@/',
        img_norm_cfg=img_norm_cfg,
        train=True),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val_map.txt',
        img_prefix='val.zip@/',
        img_norm_cfg=img_norm_cfg,
        train=False),
)
# optimizer
optimizer = dict(type='SGD', lr=0.1*data['imgs_per_gpu']/64, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
total_epochs = 120
# learning policy
lr_config = dict(
    policy='cosine',
    warmup='linear',
    warmup_iters=25600,
    warmup_ratio=0.01,
    t_max=total_epochs,
    )
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
device_ids = range(4)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/r50_120ep'
load_from = None
resume_from = None
auto_resume = False
workflow = [('train', 1), ('val', 1)]
