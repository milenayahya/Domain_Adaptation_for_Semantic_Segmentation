from torchvision.transforms import v2
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomPerspective
import random


# these are to be applied only to images, as labels are categoricla data,
# meaning they do not depend on colors/textures
bright_t = v2.ColorJitter(brightness=0.2)
contrast_t = v2.ColorJitter(contrast=0.1)
saturation_t = v2.ColorJitter(saturation=0.1)
hue_t = v2.ColorJitter(hue=0.2)
gs_t = v2.Grayscale(3)
blur_t = v2.GaussianBlur(kernel_size=15, sigma=(0.3, 0.7))
sol_t = v2.RandomSolarize(p=1, threshold=0.4)


# random horizontal flips, random rotations, and random perspective need to be applied to the image and its
# corresponding label in the same exact way, so label will continue to represent the portion of the image
# we are considering
class SyncTransform:
    def __init__(self):
        # define the probability of applying the transformations
        self.probability = 0.5
        # define parameters for rotation and perspective
        self.degrees = 90
        self.distortion_scale = 0.5

    def __call__(self, image, label):

        if random.random() < self.probability:
            image = TF.hflip(image)
            label = TF.hflip(label)

        if random.random() < self.probability:
            angle = random.uniform(-self.degrees, self.degrees)
            # rotate with same angle
            image = TF.rotate(image, angle)
            label = TF.rotate(label, angle)

        if random.random() < self.probability:
            width, height = image.size
            startpoints, endpoints = RandomPerspective.get_params(
                width, height, self.distortion_scale
            )
            # apply same perpesctive with same start and end points
            image = TF.perspective(image, startpoints, endpoints)
            label = TF.perspective(label, startpoints, endpoints)

        return image, label


def augment(image, label):

    # apply geometric transformations to image and label in a synchronized way
    transform = SyncTransform()
    transformed_image, transformed_label = transform(image, label)

    # apply color and texture transformations to image only
    t1 = v2.Compose([contrast_t, bright_t])
    t2 = v2.Compose([sol_t, gs_t, blur_t])
    t3 = v2.Compose([hue_t, saturation_t])

    if random.random() > 0.5:
        transformed_image = t1(transformed_image)

    if random.random() > 0.8:
        transformed_image = t2(transformed_image)

    if random.random() > 0.5:
        transformed_image = t3(transformed_image)

    return transformed_image, transformed_label
