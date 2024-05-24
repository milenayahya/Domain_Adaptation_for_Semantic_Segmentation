from abc import ABC  # Abstract Base Class
from typing import Callable, List, Sequence
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import v2
import torch
import numpy as np
import random

OurImageT = torch.Tensor
OurLabelT = torch.Tensor

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class BaseCustomTransformation(ABC):
    def __init__(self):
        pass

    def __call__(
        self, image: "OurImageT", label: "OurLabelT"
    ) -> tuple["OurImageT", "OurLabelT"]:
        return image, label


# the following classes inherit from the Base Class so they all have a common interface


class OurResize(BaseCustomTransformation):
    def __init__(self, size: Sequence[int]):
        self.size = size

    def __call__(self, image, label):
        return F.resize(image, list(self.size)), F.resize(
            label, list(self.size), interpolation=F.InterpolationMode.NEAREST
        )


class OurNormalization(BaseCustomTransformation):
    def __init__(self, mean: Sequence[float] = IMAGENET_MEAN, std: Sequence[float] = IMAGENET_STD):
        self.mean = mean
        self.std = std

    def __call__(self, image, label):
        return F.normalize(image, list(self.mean), list(self.std)), label


class OurRandomCrop(BaseCustomTransformation):
    def __init__(self, size: Sequence[int]):
        self.size = size

    def __call__(self, image, label):
        i, j, h, w = v2.RandomCrop(self.size).get_params(image, self.size)
        return F.crop_image(image, i, j, h, w), F.crop_mask(label, i, j, h, w)


class OurToTensor(BaseCustomTransformation):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, target_type="uint8"):
        self.target_type = target_type

    def __call__(
        self, image: OurImageT, label: OurLabelT
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])(
            image
        ), F.pil_to_tensor(label)


class OurCompose(BaseCustomTransformation):

    def __init__(self, transforms: list[Callable]):
        self.transforms = [] if transforms is None else transforms

    def __call__(self, img: OurImageT, lbl: OurLabelT) -> tuple[OurImageT, OurLabelT]:
        for t in self.transforms:
            img, lbl = t(img, lbl)
        return img, lbl


# random horizontal flips, random rotations, and random perspective need to be applied to the image and its
# corresponding label in the same exact way, so label will continue to represent the portion of the image
# we are considering
class OurGeometricAugmentationTransformations(BaseCustomTransformation):
    def __init__(self, probability=0.5, degrees=90, distortion_scale=0.5):
        # define the probability of applying the transformations
        self.probability = probability
        # define parameters for rotation and perspective
        self.degrees = degrees
        self.distortion_scale = distortion_scale

    def __call__(self, image, label):

        if random.random() < self.probability:
            image = F.hflip(image)
            label = F.hflip(label)

        if random.random() < self.probability:
            angle = random.uniform(-self.degrees, self.degrees)
            # rotate with same angle
            image = F.rotate(image, angle)
            label = F.rotate(label, angle)

        return image, label


class OurColorJitterTransformation(BaseCustomTransformation):
    def __init__(
        self,
        grayscale: float = 3,
        brightness: float = 0.2,
        hue: float = 0.2,
        saturation: float = 0.1,
        contrast: float = 0.1,
        solarize_p: float = 1,
        solarize_threshold: float = 0.4,
        blur_kernel: int = 15,
        blur_sigma: list[float] = [0.3, 0.7],
    ):
        self.grayscale = grayscale
        self.brightness = brightness
        self.hue = hue
        self.saturation = saturation
        self.contrast = contrast
        self.solarize_p = solarize_p
        self.solarize_threshold = solarize_threshold
        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma

    def __call__(self, image, label):
        bright_t = v2.ColorJitter(brightness=self.brightness)
        contrast_t = v2.ColorJitter(contrast=self.contrast)
        saturation_t = v2.ColorJitter(saturation=self.saturation)
        hue_t = v2.ColorJitter(hue=self.hue)
        gs_t = v2.Grayscale(3)
        blur_t = v2.GaussianBlur(kernel_size=self.blur_kernel, sigma=self.blur_sigma)
        sol_t = v2.RandomSolarize(p=1, threshold=0.4)

        # apply color and texture transformations to image only
        t1 = [contrast_t, bright_t]
        t2 = [sol_t, gs_t, blur_t]
        t3 = [hue_t, saturation_t]

        applied_transformation_list = []
        if random.random() > 0.5:
            applied_transformation_list.extend(t1)

        if random.random() > 0.8:
            applied_transformation_list.extend(t2)

        if random.random() > 0.5:
            applied_transformation_list.extend(t3)

        if len(applied_transformation_list) == 0:
            return image, label
        else:
            return v2.Compose(applied_transformation_list)(image), label
