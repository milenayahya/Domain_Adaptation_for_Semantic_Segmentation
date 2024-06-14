from genericpath import isdir
import json
from pathlib import Path
from pprint import pprint
from typing import Literal, Union
import torch.nn as nn
import torch
from torch.nn import functional as F
from PIL import Image
import numpy as np
import numpy.typing as npt
import pandas as pd
import random
import numbers
import torchvision
import os
import logging
from train.options.train_options import TrainOptions

logger = logging.getLogger(__name__)


def poly_lr_scheduler(
    optimizer, init_lr, iter, lr_decay_iter=1, max_iter=300, power=0.9
):
    """Polynomial decay of learning rate
    :param init_lr is base learning rate
    :param iter is a current iteration
    :param lr_decay_iter how frequently decay occurs, default is 1
    :param max_iter is number of maximum iterations
    :param power is a polymomial power

    """
    # if iter % lr_decay_iter or iter > max_iter:
    # 	return optimizer

    lr = init_lr * (1 - iter / max_iter) ** power
    optimizer.param_groups[0]["lr"] = lr
    return lr
    # return lr


def get_label_info(csv_path):
    # return label -> {label_name: [r_value, g_value, b_value, ...}
    ann = pd.read_csv(csv_path)
    label = {}
    for iter, row in ann.iterrows():
        label_name = row["name"]
        r = row["r"]
        g = row["g"]
        b = row["b"]
        class_11 = row["class_11"]
        label[label_name] = [int(r), int(g), int(b), class_11]
    return label


def one_hot_it(label, label_info):
    # return semantic_map -> [H, W]
    semantic_map = np.zeros(label.shape[:-1])
    for index, info in enumerate(label_info):
        color = label_info[info]
        # colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
        equality = np.equal(label, color)
        class_map = np.all(equality, axis=-1)
        semantic_map[class_map] = index
        # semantic_map.append(class_map)
    # semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map


def one_hot_it_v11(label, label_info):
    # return semantic_map -> [H, W, class_num]
    semantic_map = np.zeros(label.shape[:-1])
    # from 0 to 11, and 11 means void
    class_index = 0
    for index, info in enumerate(label_info):
        color = label_info[info][:3]
        class_11 = label_info[info][3]
        if class_11 == 1:
            # colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
            equality = np.equal(label, color)
            class_map = np.all(equality, axis=-1)
            # semantic_map[class_map] = index
            semantic_map[class_map] = class_index
            class_index += 1
        else:
            equality = np.equal(label, color)
            class_map = np.all(equality, axis=-1)
            semantic_map[class_map] = 11
    return semantic_map


def one_hot_it_v11_dice(label, label_info):
    # return semantic_map -> [H, W, class_num]
    semantic_map = []
    void = np.zeros(label.shape[:2])
    for index, info in enumerate(label_info):
        color = label_info[info][:3]
        class_11 = label_info[info][3]
        if class_11 == 1:
            # colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
            equality = np.equal(label, color)
            class_map = np.all(equality, axis=-1)
            # semantic_map[class_map] = index
            semantic_map.append(class_map)
        else:
            equality = np.equal(label, color)
            class_map = np.all(equality, axis=-1)
            void[class_map] = 1
    semantic_map.append(void)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float)
    return semantic_map


def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.

    # Arguments
            image: The one-hot format image

    # Returns
            A 2D array with the same width and height as the input, but
            with a depth size of 1, where each pixel value is the classified
            class key.
    """
    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,1])

    # for i in range(0, w):
    #     for j in range(0, h):
    #         index, value = max(enumerate(image[i, j, :]), key=operator.itemgetter(1))
    #         x[i, j] = index
    image = image.permute(1, 2, 0)
    x = torch.argmax(image, dim=-1)
    return x


def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.

    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """

    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,3])
    # colour_codes = label_values
    # for i in range(0, w):
    #     for j in range(0, h):
    #         x[i, j, :] = colour_codes[int(image[i, j])]
    label_values = [
        label_values[key][:3] for key in label_values if label_values[key][3] == 1
    ]
    label_values.append([0, 0, 0])
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x


def compute_global_accuracy(pred, label):
    pred = pred.flatten()
    label = label.flatten()
    total = len(label)
    count = 0.0
    for i in range(total):
        if pred[i] == label[i]:
            count = count + 1.0
    return float(count) / float(total)


def fast_compute_global_accuracy(pred: npt.NDArray, label: npt.NDArray):
    pred = pred.flatten()
    label = label.flatten()
    accuracy = (pred == label).sum() / len(label)
    return accuracy


def fast_hist(a, b, n):
    """
    a and b are predict and mask respectively
    n is the number of classes
    """
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


def per_class_iu(hist):
    epsilon = 1e-5
    return (np.diag(hist)) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)


class RandomCrop(object):
    """Crop the given PIL Image at a random location.

    Args:
            size (sequence or int): Desired output size of the crop. If size is an
                    int instead of sequence like (h, w), a square crop (size, size) is
                    made.
            padding (int or sequence, optional): Optional padding on each border
                    of the image. Default is 0, i.e no padding. If a sequence of length
                    4 is provided, it is used to pad left, top, right, bottom borders
                    respectively.
            pad_if_needed (boolean): It will pad the image if smaller than the
                    desired size to avoid raising an exception.
    """

    def __init__(self, size, seed, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.seed = seed

    @staticmethod
    def get_params(img, output_size, seed):
        """Get parameters for ``crop`` for a random crop.

        Args:
                img (PIL Image): Image to be cropped.
                output_size (tuple): Expected output size of the crop.

        Returns:
                tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        random.seed(seed)
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
                img (PIL Image): Image to be cropped.

        Returns:
                PIL Image: Cropped image.
        """
        if self.padding > 0:
            img = torchvision.transforms.functional.pad(img, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = torchvision.transforms.functional.pad(
                img, (int((1 + self.size[1] - img.size[0]) / 2), 0)
            )
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = torchvision.transforms.functional.pad(
                img, (0, int((1 + self.size[0] - img.size[1]) / 2))
            )

        i, j, h, w = self.get_params(img, self.size, self.seed)

        return torchvision.transforms.functional.crop(img, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, padding={1})".format(
            self.size, self.padding
        )


def cal_miou(miou_list, csv_path):
    # return label -> {label_name: [r_value, g_value, b_value, ...}
    ann = pd.read_csv(csv_path)
    miou_dict = {}
    cnt = 0
    for iter, row in ann.iterrows():
        label_name = row["name"]
        class_11 = int(row["class_11"])
        if class_11 == 1:
            miou_dict[label_name] = miou_list[cnt]
            cnt += 1
    return miou_dict, np.mean(miou_list)


class OHEM_CrossEntroy_Loss(nn.Module):
    def __init__(self, threshold, keep_num):
        super(OHEM_CrossEntroy_Loss, self).__init__()
        self.threshold = threshold
        self.keep_num = keep_num
        self.loss_function = nn.CrossEntropyLoss(reduction="none")

    def forward(self, output, target):
        loss = self.loss_function(output, target).view(-1)
        loss, loss_index = torch.sort(loss, descending=True)
        threshold_in_keep_num = loss[self.keep_num]
        if threshold_in_keep_num > self.threshold:
            loss = loss[loss > self.threshold]
        else:
            loss = loss[: self.keep_num]
        return torch.mean(loss)


def group_weight(weight_group, module, norm_layer, lr):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, norm_layer) or isinstance(m, nn.GroupNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    weight_group.append(dict(params=group_decay, lr=lr))
    weight_group.append(dict(params=group_no_decay, weight_decay=0.0, lr=lr))
    return weight_group


def save_checkpoint(state: dict, base_path: Union[str, Path], is_best: bool):
    os.makedirs(base_path, exist_ok=True)
    filename = os.path.join(base_path, ("latest.tar" if not is_best else "best.tar"))
    torch.save(state, filename)
    textable = dict()
    for k, v in state.items():
        if k in ["state_dict", "arch", "optimizer"]:
            # ignore these
            continue
        try:
            s = str(v)
            textable[k] = s[0:100]  # If it's longer than 100 maybe we don't need it
        except:
            pass
    with open(filename + ".json", "w") as f:
        json.dump(textable, f, indent=4)
    logger.info("Saved checkpoint")


# use models trained with FDA to generate pseudo-labels for a target dataset
def createCityscapesPseudoLabels(
    args: "TrainOptions",
    models_paths: list[str],
    split: Literal["train", "val"] = "train",
):
    # Tweaked from https://github.com/YanchaoYang/FDA/blob/master/getSudoLabel_multi.py
    from tqdm import tqdm
    from model.model_stages import BiSeNet
    from Datasets.cityscapes_torch import Cityscapes, CITYSCAPES_BASE_PATH
    import torch
    from torch.utils.data import DataLoader
    from Datasets.transformations import (
        OurCompose,
        OurNormalization,
        OurToTensor,
    )
    from PIL.Image import Resampling

    assert len(models_paths) == 3, "Must specify exactly 3 models"

    # convert list of string models passed as parameter to list of cuda models
    models = []
    for model_path in models_paths:
        # loading the models trained with different betas
        checkpoint = torch.load(model_path)
        model = BiSeNet(
            backbone=args.backbone,
            n_classes=args.num_classes,
            pretrain_model=str(args.pretrain_path),
            use_conv_last=args.use_conv_last,
        )
        if torch.cuda.is_available() and args.use_gpu:
            model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        models.append(model)
        name, epoch = checkpoint["name"], checkpoint["epoch"]
        print(f"Loaded model from {model_path}, {name=} at epoch {epoch}")

    # the loaded models:
    model1, model2, model3 = models[0], models[1], models[2]

    cityscapes_dataset = Cityscapes(
        mode=split, transforms=OurCompose([OurToTensor(), OurNormalization()])
    )
    # Create a folder to save the pseudo-labels
    PSEUDO_LABELS_PATH = os.path.join(CITYSCAPES_BASE_PATH, "pseudo", split)
    os.makedirs(PSEUDO_LABELS_PATH, exist_ok=True)

    dataloader = DataLoader(cityscapes_dataset, 1, shuffle=False)
    
    # initialize arrays to save the pseudo-labels
    predicted_label = []
    predicted_prob = []
    image_name = []


    # MBT adaptation
    # iterate over the target dataset

    with torch.no_grad():
        for i, data in tqdm(
            enumerate(dataloader),
            desc="Evaluating Pseudo Labels",
            unit="img",
            total=len(cityscapes_dataset),
        ):
            # load image and label to GPU
            image, label = data
            if torch.cuda.is_available() and args.use_gpu:
                image, label = image.cuda(), label.cuda()

            # forward pass using all 3 models
            output1 = model1(image)[0]
            output1 = nn.functional.softmax(output1, dim=1)

            output2 = model2(image)[0]
            output2 = nn.functional.softmax(output2, dim=1)

            output3 = model3(image)[0]
            output3 = nn.functional.softmax(output3, dim=1)

            # compute mean prediction
            a, b = 0.3333, 0.3333
            output = a * output1 + b * output2 + (1.0 - a - b) * output3

            output = (
                output
                .cpu()
                .data[0]
                .numpy()
            )
            output = output.transpose(1, 2, 0)

            # find the predicted label and predicted probability (the output is a softmax, so a probability)
            label, prob = np.argmax(output, axis=2), np.max(output, axis=2)
            predicted_label.append(label.copy())
            predicted_prob.append(prob.copy())

            # get the image name
            image_name.append(cityscapes_dataset.images[i])
        thres = []

    # compute the threshold for EACH class
    for i in range(args.num_classes):
        # predictions of class i
        x = predicted_prob[predicted_label == i]
        if len(x) == 0:
            thres.append(0)
            continue
        x = np.sort(x)
        # calculate threshold
        thres.append(x[int(np.round(len(x) * 0.66))])
    thres = np.array(thres)
    thres[thres > 0.9] = 0.9

    for index in tqdm(range(len(image_name)), desc="Saving Pseudo Labels", unit="img"):
        name = image_name[index]
        label = predicted_label[index]
        prob = predicted_prob[index]

        # discard predictions for classes that have probability < threshold_class (filter out the low-confidence predictions)
        for i in range(args.num_classes):
            label[(prob < thres[i]) * (label == i)] = 255
        
        # save the pseudo-label
        output = np.asarray(label, dtype=np.uint8)

        # transform array to Image, name image, and save in correct folder
        output = Image.fromarray(output)
        output_full = output.resize((2048, 1024), resample=Resampling.NEAREST)
        folder = os.path.basename(os.path.dirname(name))
        os.makedirs(os.path.join(PSEUDO_LABELS_PATH, folder), exist_ok=True)
        name = os.path.basename(name)
        name = name.replace("_leftImg8bit", "_pseudo_labelTrainIds")
        name = os.path.join(PSEUDO_LABELS_PATH, folder, name)

        # save colored pseudo-label
        colored = Image.fromarray(Cityscapes.decode(label).astype("uint8"))
        colored_full = colored.resize((2048, 1024), resample=Resampling.NEAREST)
        name_color = name.replace("_pseudo_labelTrainIds", "_pseudo_color")
        output_full.save(name)
        colored_full.save(name_color)


def visualizer(args: "TrainOptions", model_path: str, save_path: str, image_index = 0, split: Literal['train', 'val'] = "train"):
    from model.model_stages import BiSeNet
    from Datasets.cityscapes_torch import Cityscapes
    import torch
    from torch.utils.data import DataLoader
    from Datasets.transformations import (
        OurCompose,
        OurNormalization,
        OurToTensor,
    )
    import numpy as np

    checkpoint = torch.load(model_path)
    model = BiSeNet(
        backbone=args.backbone,
        n_classes=args.num_classes,
        pretrain_model=str(args.pretrain_path),
        use_conv_last=args.use_conv_last,
    )
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    os.makedirs(save_path, exist_ok=True)

    pseudo_dataset = Cityscapes(
        mode=split, transforms=OurCompose([OurToTensor(), OurNormalization()])
    )

    dataloader = DataLoader(pseudo_dataset, 1, shuffle=False)
    output: torch.Tensor
    for i, data in enumerate(dataloader):
        if i != image_index:
            continue
        img, _ = data
        output = model(img)[0]
        break
    out_np: npt.NDArray = (
        output
        .cpu()
        .data[0]
        .numpy()
    )
    out_np = out_np.transpose(1, 2, 0)
    out_np = np.argmax(out_np, axis=2)
    colored = Image.fromarray(Cityscapes.decode(out_np).astype("uint8"))
    # print("Saving Pred")
    colored.save(os.path.join(save_path, "pred.png"))


if __name__ == "__main__":
    from tqdm import tqdm
    args = TrainOptions.default()
    models = ["2B/SGD-4", "2C2/SGD-6", "3/SGD-6-aug-B", "3/SGD-4-normal-B", "FDA/SGD-6-01-AUG", "FDA/SGD-6-01-SST-AUG-FINAL"]

    # postfix = "best.tar" if args.use_best else "latest.tar"
    # models = [os.path.join(args.save_model_path, "FDA", it, postfix) for it in models]
    # print(f"{models=}")
    # if any([not os.path.isfile(it) for it in models]):
    #     print("Some models are not found")
    #     exit(1)
    # createCityscapesPseudoLabels(args, models, split="train")
    for model in models:
        model_path = os.path.abspath(os.path.join(args.save_model_path, model))
        if not os.path.exists(model_path):
            raise Exception("Bad Model name", model_path)
        
    for model in tqdm(models, unit="model"):
        model_path = os.path.abspath(os.path.join(args.save_model_path, model))
        visualizer(args, os.path.join(model_path, "best.tar"), os.path.join(args.save_model_path, "..", 'logs', "visual", model), 0, "val")
