import time
# model settings
model = dict(
    type='VanillaNet',
    backbone=dict(
        type='ScaleNet',
        depth=34,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        develop=dict(
            skip_first=True,
            gumbel=dict(
                decay_method='exp',
                max_T=1.0,
                decay_alpha=0.999988,
            ),
            mask_kernel=3,
            mask_prob=0.9526,
            mask_type='hard_inference',
            mask_hard_threshold=0.5,
            grid_conv=False,
            grid_mask=True,
            grid_interval=5,
            interpolate='lg',
            lg_kernel=5,
            lg_init_sigma=3.0,
            lg_type='reciprocal',
            backbone_loss_weight=0.1,
            loss_position='gumble_sigmoid',
        )
    ),
    cls_head=dict(
        type='FCHead',
        num_classes=1000))

train_cfg = dict(
)
test_cfg = dict(
    develop=dict(
        gumbel=dict(
            decay_method='exp',
            max_T=0.01,
            decay_alpha=1.0,
        )
    )
)
# dataset settings
dataset_type = 'ImageNetDataset'
# data_root = '/hdfs/public/imagenet/new/'
data_root = '/blob/data/imagenet/'
# data_root = '/data/home/zhez/mnt/vc_image_recog/v-zhxia/Data/ImageNet-Zip/'
# data_root = '/mnt/zhez/imagenet_zip/'
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
        # ann_file='val_map.txt',
        # img_prefix='val.zip@/',
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
optimizer = dict(type='SGD', lr=0.1 * data['imgs_per_gpu']/64, momentum=0.9, weight_decay=0.0001)
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
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
auto_resume = False
workflow = [('train', 1), ('val', 1)]

def _dir_parser(item):
    if isinstance(item, dict):
        _dir_out = []
        for key in item.keys():
            _dir_out.append(_dir_parser(item[key]))
        return '_'.join(_dir_out)
    elif isinstance(item, list) or isinstance(item, tuple):
        return ''.join([str(i) for i in item])
    elif isinstance(item, str):
        return item
    elif isinstance(item, int):
        return str(item)
    elif isinstance(item, float):
        if int(item) == item:
            return str(int(item))
        return ''.join(str(item).split('.'))
    else:
        raise NotImplementedError

develop = dict(
    model=model['backbone']['type'],
    depth=model['backbone']['depth'],
    model_develop=model['backbone']['develop'],
    epoch=str(total_epochs) + 'ep',
    time=time.strftime('%b%d_%H%M', time.localtime()),
)

# Get Work Directory
_dir = []
_dir.append(_dir_parser(develop))
work_dir = f'/blob/data/imagenet_work_dirs/'
# work_dir = './imagenet_work_dirs/'
work_dir += '_'.join(_dir)