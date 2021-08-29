import os
import torchvision.transforms as transforms
from PIL import Image
from .dataset_folder import ImageFolder


class ImageNetDataset(ImageFolder):

    def __init__(self, data_root, img_norm_cfg, resize_scale=256, crop_scale=224,
                 ann_file='', img_prefix='', train=True, cache_mode=False):
        if train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(crop_scale),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(**img_norm_cfg),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(resize_scale),
                transforms.CenterCrop(crop_scale),
                transforms.ToTensor(),
                transforms.Normalize(**img_norm_cfg),
            ])
        super(ImageNetDataset, self).__init__(root=data_root, ann_file=ann_file, img_prefix=img_prefix,
                                              transform=transform, cache_mode=cache_mode)

    def __getitem__(self, index):
        sample, target = super(ImageNetDataset, self).__getitem__(index)
        data = {'img': sample,
                'gt_label': target}

        return data

