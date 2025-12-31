# --------------------------------------------------------
# References:
# DinoV2: https://github.com/facebookresearch/dinov2
# --------------------------------------------------------

import numpy as np
from typing import Sequence

import torch
from torchvision import transforms


class GaussianBlur(transforms.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = transforms.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)


class MaybeToTensor(transforms.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)


# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)


# This roughly matches torchvision's preset for classification training:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L6-L44
def make_classification_train_transform(
    *,
    crop_size: int = 224,
    interpolation=transforms.InterpolationMode.BICUBIC,
    hflip_prob: float = 0.5,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
):
    transforms_list = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
    if hflip_prob > 0.0:
        transforms_list.append(transforms.RandomHorizontalFlip(hflip_prob))
    transforms_list.extend(
        [
            MaybeToTensor(),
            make_normalize_transform(mean=mean, std=std),
        ]
    )
    return transforms.Compose(transforms_list)


# This matches (roughly) torchvision's preset for classification evaluation:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L47-L69
def make_classification_eval_transform(
    *,
    resize_size: int = 256,
    interpolation=transforms.InterpolationMode.BICUBIC,
    crop_size: int = 224,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    transforms_list = [
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
        MaybeToTensor(),
        make_normalize_transform(mean=mean, std=std),
    ]
    return transforms.Compose(transforms_list)


class SentinelNormalize:
    """
    Normalization for Sentinel-2 imagery, inspired from
    https://github.com/ServiceNow/seasonal-contrast/blob/8285173ec205b64bc3e53b880344dd6c3f79fa7a/datasets/bigearthnet_dataset.py#L111
    """

    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, x, *args, **kwargs):
        min_value = self.mean - 2 * self.std
        max_value = self.mean + 2 * self.std
        img = (x - min_value) / (max_value - min_value) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img


class SentinelClassificationTransform:
    def __init__(
        self,
        crop_size=96,
        interpolation=transforms.InterpolationMode.BICUBIC,
        hflip_prob: float = 0.5,
        dataset_mean=None,
        dataset_std=None,
        is_train=True,
    ):
        # self.resize_size = resize_size
        self.crop_size = crop_size
        self.interpolation = interpolation
        self.hflip_prob = hflip_prob
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std
        self.is_train = is_train

        self.transform = self.create_transforms()

    def create_transforms(self):
        transforms_list = []
        if self.is_train:
            transforms_list.extend(
                [
                    SentinelNormalize(self.dataset_mean, self.dataset_std),
                    transforms.ToTensor(),
                    transforms.RandomResizedCrop(self.crop_size, interpolation=self.interpolation),
                ]
            )
            if self.hflip_prob > 0:
                transforms_list.append(transforms.RandomHorizontalFlip(self.hflip_prob))

        else:
            if self.crop_size <= 224:
                crop_pct = 224 / 256
            else:
                crop_pct = 1.0
            size = int(self.crop_size / crop_pct)

            transforms_list = [
                SentinelNormalize(self.dataset_mean, self.dataset_std),
                transforms.ToTensor(),
                transforms.Resize(size),
                transforms.CenterCrop(self.crop_size),
            ]

        return transforms.Compose(transforms_list)

    def reset_normalize_transform(self, mean, std):
        # This function is essential for make_dataset to set the correct sentinel normalize
        self.dataset_mean = mean
        self.dataset_std = std
        self.transform = self.create_transforms()

    def __call__(self, x):
        return self.transform(x)


def get_transform(dataset_path, is_train=True):
    if "sentinel" in dataset_path:
        return SentinelClassificationTransform(is_train=is_train)

    return make_classification_train_transform() if is_train else make_classification_eval_transform()
