import os
import torchvision.transforms as transforms
from PIL import Image
import torchvision.datasets as datasets


class Cifar10Dataset(datasets.CIFAR10):

    def __init__(self, data_root, train, img_norm_cfg, ann_file=None, img_prefix=None):
        if train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(**img_norm_cfg)
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(**img_norm_cfg)
            ])
        super(Cifar10Dataset, self).__init__(root=data_root, train=train, transform=transform, download=True)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = super(Cifar10Dataset, self).__getitem__(index)
        data = {'img': img,
                'gt_label': target}

        return data
