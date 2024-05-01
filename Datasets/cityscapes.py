#!/usr/bin/python
# -*- encoding: utf-8 -*-
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as v2
import torchvision.transforms.functional as TF
from pathlib import Path
from collections import namedtuple
import numpy as np
import os.path
import random
from Datasets import CITYSCAPES_BASE_PATH
from tqdm import tqdm
from typing import Literal

class CityScapes(Dataset):
    def __init__(self, mode, cropSize=(512, 1024), load_mode: Literal["instant", "on_request"] = "on_request"):
        super(CityScapes, self).__init__()

        self.mode = mode
        self.load_mode = load_mode
        self.cropSize = cropSize
        # self.root = Path("/content/Cityscapes/Cityscapes/Cityspaces")  #google colab path
        self.root = Path(CITYSCAPES_BASE_PATH)  # local path
        # self.root = Path("./Cityscapes/Cityscapes/Cityspaces")   #local path
        # self.root = Path("./Cityscapes/Cityscapes/Cityspaces")   #local path
        # self.root = Path("./Cityscapes/Cityscapes/Cityspaces")   #local path

        if mode == "train":
            self.images_path = self.root / "images/train"
            self.labels_path = self.root / "gtFine/train"
            self.dirs = [
                "hanover",
                "jena",
                "krefeld",
                "monchengladbach",
                "strasbourg",
                "stuttgart",
                "tubingen",
                "ulm",
                "weimar",
                "zurich",
            ]

        if mode == "val":
            self.images_path = self.root / "images/val"
            self.labels_path = self.root / "gtFine/val"
            self.dirs = ["frankfurt", "lindau", "munster"]

        print("Checking paths:")
        print("Images path:", self.images_path)
        print("Labels path:", self.labels_path)
        print("Directories:", self.dirs)

        # mean and std of ImageNet dataset
        self.transform = v2.Compose(
            [
                # v2.ToTensor(),
                # v2.ToTensor(),
                v2.ToTensor(),
                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        self.samples = []
        self.images = []
        self.labels = []
        self.image_filenames = []
        self.label_filenames = []

        for dir_name in tqdm(
            self.dirs, desc=f"Parsing Cityscapes ({mode})", unit="dir", leave=False
        ):

            img_dir_path = self.images_path / dir_name
            label_dir_path = self.labels_path / dir_name
            img_files = sorted(img_dir_path.glob("*.png"))
            label_files = sorted(label_dir_path.glob("*labelTrainIds.png"))

            self.image_filenames += img_files
            self.label_filenames += label_files

            if self.load_mode == "instant":
                for img_path, label_path in zip(img_files, label_files):
                    img_tensor, label_tensor = self.read_image(img_path, label_path)
            

        assert (
            len(self.image_filenames) > 0
        ), f"Seems like Dataset is Missing {self.images_path=} {self.root=}"

        # Create tuples of (image, label) and append to samples
        self.samples.extend(zip(self.images, self.labels))
        print(f"Cityscapes {mode} dataset initialized with {len(self.image_filenames)} images ({self.load_mode})")

    def map_labels(self, label):
        # we vectorize the get function of a dictionary since we want to
        # pass as input a label image, which is an array of labelIds
        mapped_labels = np.vectorize(id_to_trainId.get)(
            label, 255
        )  # Use 255 as default for ignored labels
        return mapped_labels

    def read_image(self, img_path: str, label_path: str) -> tuple:
        img_tensor, label_tensor = None, None
        with Image.open(img_path).convert("RGB") as img:
            if self.mode == "train":
                # i,j,h,w = v2.RandomCrop.get_params(img, cropSize)
                # img = TF.crop(img,i,j,h,w)
                img = TF.resize(img, self.cropSize)
            img_tensor = self.transform(img)

        with Image.open(label_path) as label:

            # crop label in same position as image
            if self.mode == "train":
                # label= TF.crop(label,i,j,h,w)
                label = TF.resize(label, self.cropSize)
            label_array = np.array(label)
            label_mapped = self.map_labels(label_array)

            label_tensor = torch.from_numpy(
                label_mapped
            ).long()  # to perserve int format
            label_tensor = label_tensor.unsqueeze(0)

        return img_tensor, label_tensor

    def __getitem__(self, idx):
        if self.load_mode == "instant":
            image = self.images[idx]
            label = self.labels[idx]
            return image, label
        else:
            img_path, label_path = self.image_filenames[idx], self.label_filenames[idx]
            return self.read_image(img_path, label_path)
    def __len__(self):
        return len(self.image_filenames)

Label = namedtuple(
    "Label",
    [
        "name",
        "id",
        "trainId",
        "category",
        "categoryId",
        "hasInstances",
        "ignoreInEval",
        "color",
    ],
)

# fmt: off
labels_dict = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]
# fmt: on

id_to_trainId = {
    label.id: label.trainId
    for label in labels_dict
    if label.trainId != 255 and label.trainId != -1
}


if __name__ == "__main__":

    train_dataset = CityScapes("train")
    val_dataset = CityScapes("val")
