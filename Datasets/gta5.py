from typing import Callable, Literal, Union
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import numpy as np

from .transformations import OurCompose, OurToTensor
from .augmentation import augment
from . import GTA5_BASE_PATH
import logging

logger = logging.getLogger(__name__)
# run splitGTA5.py before to split the data into train and val


class GTA5(Dataset):
    mode: Literal["train", "val"]
    load_mode: Literal["instant", "on_request"]
    transforms: Callable

    def __init__(
        self,
        mode: Literal["train", "val"],
        load_mode: Literal["instant", "on_request"] = "on_request",
        transforms=None,
    ):
        super(GTA5, self).__init__()

        self.mode = mode
        self.load_mode = load_mode

        # self.root = Path("/content/GTA5/GTA5")  #google colab path
        self.root = Path(GTA5_BASE_PATH)  # local path

        if mode == "train":
            self.images_path = self.root / "images/train"
            self.labels_path = self.root / "labels/train"

        if mode == "val":
            self.images_path = self.root / "images/val"
            self.labels_path = self.root / "labels/val"

        logger.info("Images path: %s" % self.images_path)
        logger.info("Labels path: %s" % self.labels_path)

        self.transforms = OurCompose([]) if transforms is None else transforms

        # fmt: off
        self.id_to_trainid = {
            7: 0,  8: 1,  11: 2,  12: 3,  13: 4,  17: 5,  19: 6, 
            20: 7,  21: 8,  22: 9,  23: 10,  24: 11,  25: 12,  26: 13, 
            27: 14,  28: 15,  31: 16,  32: 17,  33: 18, 
        }
        # fmt: on

        self.images = []
        self.labels = []

        self.image_filenames = sorted(self.images_path.glob("*.png"))
        self.label_filenames = sorted(self.labels_path.glob("*.png"))

        if self.load_mode == "instant":
            for img_path, label_path in zip(self.image_filenames, self.label_filenames):
                img_tensor, label_tensor = self.read_image(img_path, label_path)
                self.images.append(img_tensor)
                self.labels.append(label_tensor)

        assert (
            len(self.image_filenames) > 0
        ), f"Seems like Dataset is Missing {self.images_path=} {self.root=}"

        logger.info(f"DONE processing {len(self)} images and labels")

    def __getitem__(self, idx):
        if self.load_mode == "instant":
            image = self.images[idx]
            label = self.labels[idx]
            return image, label
        else:
            img_path, label_path = self.image_filenames[idx], self.label_filenames[idx]
            return self.read_image(img_path, label_path)

    def read_image(self, img_path: Union[str, Path], label_path: Union[str, Path]) -> tuple:
        with Image.open(img_path).convert("RGB") as img, Image.open(
            label_path
        ) as label:
            label = np.asarray(label, np.float32)

            label_copy = 255 * np.ones(label.shape, dtype=np.float32)
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v

            label_tensor = torch.tensor(label_copy, dtype=torch.float32)
            img_tensor, label_tensor = self.transforms(img, label_tensor)
            return img_tensor, label_tensor

    def __len__(self):
        return len(self.image_filenames)


if __name__ == "__main__":

    train_dataset = GTA5("train")
    ti, tl = train_dataset[4]
    val_dataset = GTA5("val", transforms=OurToTensor())
    vi, vl = val_dataset[4]
