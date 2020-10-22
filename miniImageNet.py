import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image

class miniImageNet(data.Dataset):

    def __init__(self, root, split):
        self.root = os.path.join(root, 'mini-imagenet', 'images')
        self.split = split
        assert (split == 'train' or split == 'val')
        self.csv_file = os.path.join(root, 'mini-imagenet', split + '_split.csv')
        self.images = []
        self.labels = []
        self.transform = None

        mean_pix = [0.485, 0.456, 0.406]
        std_pix = [0.229, 0.224, 0.225]

        if self.split == 'val':
            self.transform = transforms.Compose([
                transforms.Resize(224/0.875),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_pix, std=std_pix),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_pix, std=std_pix),
            ])

        with open(self.csv_file, 'r') as f:
            for i, line in enumerate(f):
                image_path = os.path.join(self.root, line.split(',')[0])
                label = line.split(',')[1][:-1]
                self.images.append(image_path)
                self.labels.append(label)

    def __getitem__(self, index):
        image_path = self.images[index]
        img = Image.open(image_path).convert('RGB')
        target = int(self.labels[index])

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.labels)
