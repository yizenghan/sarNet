import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch
import torchvision.transforms as transforms
from auto_augment import auto_augment_transform


def get_transform(args, is_train_set=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]) # RGB

    lighting = Lighting(alphastd=0.1,
                        eigval=[0.2175, 0.0188, 0.0045],
                        eigvec=[[-0.5675,  0.7192,  0.4009],
                                [-0.5808, -0.0045, -0.8140],
                                [-0.5836, -0.6948,  0.4203]])


    if is_train_set:
        transform_set = []
        transform_set += [transforms.RandomResizedCrop(size=args.cfg['train_cfg']['crop_size'])] # default size=224
        if args.colorjitter:
            transform_set += [transforms.ColorJitter(brightness=0.4,
                                                    contrast=0.4,
                                                    saturation=0.4)]
        transform_set += [transforms.RandomHorizontalFlip()]
        if args.autoaugment:
            # transform_set += [ImageNetPolicy()]
            transform_set += [auto_augment_transform(img_size=args.cfg['train_cfg']['crop_size'])]
        transform_set += [transforms.ToTensor()]
        if args.change_light:
            transform_set += [lighting]
        transform_set += [normalize]
        return transforms.Compose(transform_set)
    else:
        if args.cfg['test_cfg']['crop_type'] == 'resnest':
            return transforms.Compose([
                ECenterCrop(args.cfg['test_cfg']['crop_size']),
                transforms.ToTensor(),
                normalize
            ])
        elif args.cfg['test_cfg']['crop_type'] == 'normal':
            return transforms.Compose([
                        transforms.Resize(int(args.cfg['test_cfg']['crop_size']/0.875)),
                        transforms.CenterCrop(args.cfg['test_cfg']['crop_size']),
                        transforms.ToTensor(),
                        normalize,
                    ])
        elif args.cfg['test_cfg']['crop_type'] == 'tencrop':
            return transforms.Compose([
                        transforms.Resize(int(args.cfg['test_cfg']['crop_size']/0.875)),
                        transforms.TenCrop(args.cfg['test_cfg']['crop_size']),
                        TenCropToTensor(),
                        TenCropNormalize(normalize),
                    ])
        else:
            raise NotImplemented("The crop type {} is not implemented! Please select from "
                                 "[normal, resnest]".format(args.cfg['test_cfg']['crop_type']))


class Lighting(object):
    """Lighting noise (AlexNet-style PCA-based noise)

    Args:
        alphastd (float): The std of the normal distribution.
        eigval (array): The eigenvalue of RGB channel.
        eigvec (array): The eigenvec of RGB channel.

    """

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be processed.

        Returns:
            Tensor: Add lighting noise Tensor image.

        Code Reference: https://github.com/clovaai/CutMix-PyTorch/blob/master/utils.py
        """
        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))

    def __repr__(self):
        return self.__class__.__name__ + '(alphastd={0})'.format(self.alphastd)


class ECenterCrop:
    """Crop the given PIL Image and resize it to desired size.
    Args:
        img (PIL Image): Image to be cropped. (0,0) denotes the top left corner of the image.
        output_size (sequence or int): (height, width) of the crop box. If int,
            it is used for both directions
    Returns:
        PIL Image: Cropped image.
    """
    def __init__(self, imgsize):
        self.imgsize = imgsize
        self.resize_method = transforms.Resize((imgsize, imgsize))#, interpolation=PIL.Image.BICUBIC)

    def __call__(self, img):
        image_width, image_height = img.size
        image_short = min(image_width, image_height)

        crop_size = float(self.imgsize) / (self.imgsize + 32) * image_short

        crop_height, crop_width = crop_size, crop_size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        img = img.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))
        return self.resize_method(img)


class TenCropToTensor(object):

    def __init__(self):
        self.totensor = transforms.ToTensor()

    def __call__(self, imgs):

        imgs = list(imgs)
        for i in range(len(imgs)):
            imgs[i] = self.totensor(imgs[i])

        return torch.stack(imgs)


class TenCropNormalize(object):

    def __init__(self, normalize):
        self.normalize = normalize

    def __call__(self, imgs):

        for i in range(len(imgs)):
            imgs[i] = self.normalize(imgs[i])

        return imgs


class RGB2BGR(object):

    def __call__(self, img):

        tmp = img[0, :, :].clone()
        img[0, :, :] = img[2, :, :]
        img[2, :, :] = tmp

        return img


# class ImageNetPolicy(object):
#     """ Randomly choose one of the best 25 Sub-policies on ImageNet.
#         Example:
#         >>> policy = ImageNetPolicy()
#         >>> transformed = policy(image)
#         Example as a PyTorch Transform:
#         >>> transform=transforms.Compose([
#         >>>     transforms.Resize(256),
#         >>>     ImageNetPolicy(),
#         >>>     transforms.ToTensor()])
#     """
#     def __init__(self, fillcolor=(128, 128, 128)):
#         self.policies = [
#             SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9, fillcolor),
#             SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
#             SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
#             SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor),
#             SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
#
#             SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8, fillcolor),
#             SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7, fillcolor),
#             SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor),
#             SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8, fillcolor),
#             SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor),
#
#             SubPolicy(0.8, "rotate", 8, 0.4, "color", 0, fillcolor),
#             SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2, fillcolor),
#             SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor),
#             SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
#             SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),
#
#             SubPolicy(0.8, "rotate", 8, 1.0, "color", 2, fillcolor),
#             SubPolicy(0.8, "color", 8, 0.8, "solarize", 7, fillcolor),
#             SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, fillcolor),
#             SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9, fillcolor),
#             SubPolicy(0.4, "color", 0, 0.6, "equalize", 3, fillcolor),
#
#             SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
#             SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
#             SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
#             SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),
#             SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor)
#         ]
#
#
#     def __call__(self, img):
#         policy_idx = random.randint(0, len(self.policies) - 1)
#         return self.policies[policy_idx](img)
#
#     def __repr__(self):
#         return "AutoAugment ImageNet Policy"
#
#
# class SubPolicy(object):
#     def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
#         ranges = {
#             "shearX": np.linspace(0, 0.3, 10),
#             "shearY": np.linspace(0, 0.3, 10),
#             "translateX": np.linspace(0, 150 / 331, 10),
#             "translateY": np.linspace(0, 150 / 331, 10),
#             "rotate": np.linspace(0, 30, 10),
#             "color": np.linspace(0.0, 0.9, 10),
#             "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
#             "solarize": np.linspace(256, 0, 10),
#             "contrast": np.linspace(0.0, 0.9, 10),
#             "sharpness": np.linspace(0.0, 0.9, 10),
#             "brightness": np.linspace(0.0, 0.9, 10),
#             "autocontrast": [0] * 10,
#             "equalize": [0] * 10,
#             "invert": [0] * 10
#         }
#
#         func = {
#             "autocontrast": self.autocontrast,
#             "equalize": self.equalize,
#             "invert": self.invert,
#             "rotate": self.rotate,
#             "posterize": self.posterize,
#             "solarize": self.solarize,
#             ### "solarizeAdd"
#             "color": self.color,
#             "contrast": self.contrast,
#             "brightness": self.brightness,
#             "sharpness": self.sharpness,
#             "shearX": self.shearX,
#             "shearY": self.shearY,
#             "translateX": self.translateX,
#             "translateY": self.translateY,
#         }
#         self.fillcolor = fillcolor
#
#         self.p1 = p1
#         self.operation1 = func[operation1]
#         self.magnitude1 = ranges[operation1][magnitude_idx1]
#         self.p2 = p2
#         self.operation2 = func[operation2]
#         self.magnitude2 = ranges[operation2][magnitude_idx2]
#
#
#     def __call__(self, img):
#         if random.random() < self.p1: img = self.operation1(img, self.magnitude1)
#         if random.random() < self.p2: img = self.operation2(img, self.magnitude2)
#         return img
#
#     def shearX(self, img, magnitude):
#         return img.transform(
#             img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
#             Image.BICUBIC, fillcolor=self.fillcolor)
#
#     def shearY(self, img, magnitude):
#         return img.transform(
#             img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
#             Image.BICUBIC, fillcolor=self.fillcolor)
#
#     def translateX(self, img, magnitude):
#         return img.transform(
#             img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
#             fillcolor=self.fillcolor)
#
#     def translateY(self, img, magnitude):
#         return img.transform(
#             img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
#             fillcolor=self.fillcolor)
#
#     # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
#     def rotate(self, img, magnitude):
#         rot = img.convert("RGBA").rotate(magnitude)
#         return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)
#
#     def color(self, img, magnitude):
#         return ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1]))
#
#     def posterize(self, img, magnitude):
#         return ImageOps.posterize(img, magnitude)
#
#     def solarize(self, img, magnitude):
#         return ImageOps.solarize(img, magnitude)
#
#     def contrast(self, img, magnitude):
#         return ImageEnhance.Contrast(img).enhance(1 + magnitude * random.choice([-1, 1]))
#
#     def sharpness(self, img, magnitude):
#         return ImageEnhance.Sharpness(img).enhance(1 + magnitude * random.choice([-1, 1]))
#
#     def brightness(self, img, magnitude):
#         return ImageEnhance.Brightness(img).enhance(1 + magnitude * random.choice([-1, 1]))
#
#     def autocontrast(self, img, magnitude):
#         return ImageOps.autocontrast(img)
#
#     def equalize(self, img, magnitude):
#         return ImageOps.equalize(img)
#
#     def invert(self, img, magnitude):
#         return ImageOps.invert(img)