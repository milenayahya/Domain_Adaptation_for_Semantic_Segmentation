#!/usr/bin/python
# -*- encoding: utf-8 -*-
import logging
from datetime import datetime, timedelta
import math
from typing import Literal
from model.model_stages import BiSeNet
import torch
from torch.utils.data import DataLoader
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import torch.cuda.amp as amp
from tqdm import tqdm
import os
from Datasets.cityscapes import CityScapes
from Datasets.gta5 import gta5
from utils import (
    fast_compute_global_accuracy,
    reverse_one_hot,
    compute_global_accuracy,
    fast_hist,
    per_class_iu,
    save_checkpoint,
    poly_lr_scheduler,
)
from logs.tglog import RequestsHandler, LogstashFormatter, TGFilter

tg_handler = RequestsHandler()
formatter = LogstashFormatter()
filter = TGFilter()
tg_handler.setFormatter(formatter)
tg_handler.addFilter(filter)
logging.basicConfig(
    format="%(asctime)s [%(filename)s@%(funcName)s] [%(levelname)s]:> %(message)s",
    handlers=[logging.FileHandler("logs/debug.log"), logging.StreamHandler(), tg_handler],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


DatasetName = Literal["Cityscapes", "GTA5"]


def val(args, model, dataloader) -> tuple[float, float]:
    with torch.no_grad():
        model.eval()
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        pbar = tqdm(total=len(dataloader), desc="Evaluation", unit="img")
        for i, (data, label) in enumerate(dataloader):
            label = label.type(torch.LongTensor)
            data = data.cuda()
            label = label.long().cuda()

            # get RGB predict image
            predict, _, _ = model(data)
            predict = predict.squeeze(0)
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())

            # get RGB label image
            label = label.squeeze()
            label = np.array(label.cpu())

            # compute per pixel accuracy
            precision = fast_compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)

            # there is no need to transform the one-hot array to visual RGB array
            # predict = colour_code_segmentation(np.array(predict), label_info)
            # label = colour_code_segmentation(np.array(label), label_info)
            precision_record.append(precision)
            pbar.update(1)

        precision = np.mean(precision_record)
        miou_list = per_class_iu(hist)
        miou = np.mean(miou_list)
        logger.info("Validation: precision per pixel for test: %.3f" % precision)
        logger.info("Validation: mIoU for validation: %.3f" % miou)
        logger.info(f"Validation: mIoU per class:\n{miou_list}")

        return precision, miou


def train(
    args, model, optimizer, dataloader_train, dataloader_val, training_name: str = None
):
    writer = SummaryWriter(comment="{}".format(args.optimizer))

    scaler = amp.GradScaler()

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    precision, miou = 0, 0
    max_miou = 0
    step = 0
    start_epoch = 1
    checkpoint_filename = os.path.join(args.save_model_path, "latest.tar")
    runs_execution_time = []
    if args.resume and os.path.exists(checkpoint_filename):
        checkpoint = torch.load(checkpoint_filename)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        max_miou = checkpoint["max_miou"]
        logger.info(f"Loaded latest checkpoint that was at epoch n. {start_epoch}")

    tensorboard_base_name = (
        "epoch/" if training_name is None else f"{training_name}/epoch/"
    )

    for epoch in range(start_epoch, args.num_epochs + 1):
        ts_start = datetime.today()
        lr = poly_lr_scheduler(
            optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs
        )
        model.train()
        tq = tqdm(total=len(dataloader_train) * args.batch_size, unit="img")
        writer.add_scalar(f"{tensorboard_base_name}/learning_rate", lr, epoch)
        tq.set_description("epoch %d, lr %f" % (epoch, lr))
        loss_record = []
        for i, (data, label) in enumerate(dataloader_train):

            data = data.cuda()
            label = label.long().cuda()

            optimizer.zero_grad()

            with amp.autocast():
                output, out16, out32 = model(data)
                loss1 = loss_func(output, label.squeeze(1))
                loss2 = loss_func(out16, label.squeeze(1))
                loss3 = loss_func(out32, label.squeeze(1))
                loss = loss1 + loss2 + loss3

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tq.update(args.batch_size)
            tq.set_postfix(loss="%.6f" % loss)
            step += 1
            writer.add_scalar("loss_step", loss, step)
            loss_record.append(loss.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar(
            f"{tensorboard_base_name}/loss_epoch_train", float(loss_train_mean), epoch
        )
        ts_duration: timedelta = datetime.today() - ts_start
        logger.info(f"loss for train @ {epoch=}: {loss_train_mean} after {ts_duration.seconds} seconds")
        runs_execution_time.append(ts_duration.seconds)
        writer.add_scalar(f"{tensorboard_base_name}/training_duration", ts_duration.seconds, epoch)
        if epoch % args.checkpoint_step == 0:
            logger.info(f"Saving latest checkpoint @ {epoch=}, with max_miou (of latest val) = {max_miou}")
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": str(model),
                    "state_dict": model.state_dict(),
                    "max_miou": max_miou,
                    "optimizer": optimizer.state_dict(),
                },
                args.save_model_path,
                False,
            )

        if epoch % args.validation_step == 0:
            precision, miou = val(args, model, dataloader_val)
            if miou > max_miou:
                max_miou = miou
                logger.info(f"Saving best checkpoint @ {epoch=}, with {max_miou=}")
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "arch": str(model),
                        "state_dict": model.state_dict(),
                        "max_miou": max_miou,
                        "optimizer": optimizer.state_dict(),
                    },
                    args.save_model_path,
                    True,
                )
            writer.add_scalar(
                f"{tensorboard_base_name}/precision_val", precision, epoch
            )
            writer.add_scalar(f"{tensorboard_base_name}/miou val", miou, epoch)
    logger.info(f"tg:Finished training of {training_name} after {sum(runs_execution_time)/len(runs_execution_time)} seconds with a {max_miou=} and {precision=}")


def str2bool(v: str) -> bool:
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")


def parse_args():
    parse = argparse.ArgumentParser()

    parse.add_argument(
        "--mode",
        dest="mode",
        type=str,
        default="train",
    )

    parse.add_argument(
        "--backbone",
        dest="backbone",
        type=str,
        default="CatmodelSmall",
    )
    parse.add_argument(
        "--pretrain_path",
        dest="pretrain_path",
        type=str,
        default="./STDCNet813M_73.91.tar",
    )
    parse.add_argument(
        "--use_conv_last",
        dest="use_conv_last",
        type=str2bool,
        default=False,
    )
    parse.add_argument(
        "--num_epochs", type=int, default=50, help="Number of epochs to train for"
    )
    parse.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint",
    )
    parse.add_argument(
        "--checkpoint_step",
        type=int,
        default=1,
        help="How often to save checkpoints (epochs)",
    )
    parse.add_argument(
        "--validation_step",
        type=int,
        default=5,
        help="How often to perform validation (epochs)",
    )
    parse.add_argument(
        "--crop_height",
        type=int,
        default=512,
        help="Height of cropped/resized input image to modelwork",
    )
    parse.add_argument(
        "--crop_width",
        type=int,
        default=1024,
        help="Width of cropped/resized input image to modelwork",
    )
    parse.add_argument(
        "--batch_size", type=int, default=8, help="Number of images in each batch"
    )
    parse.add_argument(
        "--learning_rate", type=float, default=0.01, help="learning rate used for train"
    )
    parse.add_argument("--num_workers", type=int, default=4, help="num of workers")
    parse.add_argument(
        "--num_classes", type=int, default=19, help="num of object classes (with void)"
    )
    parse.add_argument(
        "--cuda", type=str, default="0", help="GPU ids used for training"
    )
    parse.add_argument(
        "--use_gpu", type=bool, default=True, help="whether to user gpu for training"
    )
    parse.add_argument(
        "--save_model_path", type=str, default="./results", help="path to save model"
    )
    parse.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help="optimizer, support rmsprop, sgd, adam",
    )
    parse.add_argument("--loss", type=str, default="crossentropy", help="loss function")

    return parse.parse_args()


def main(
    training_ds_name: DatasetName,
    validation_ds_name: DatasetName,
    *,
    augmentation: bool = False,
    save_model_postfix: str = "",
):
    args = parse_args()

    # dataset
    n_classes = args.num_classes

    mode = args.mode

    if training_ds_name == "Cityscapes":
        train_dataset = CityScapes(mode)
    elif training_ds_name == "GTA5" and not augmentation:
        train_dataset = gta5(mode)
    elif training_ds_name == "GTA5" and augmentation:
        train_dataset = gta5(mode, aug=True)

    dataloader_train = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
    )

    if validation_ds_name == "Cityscapes":
        val_dataset = CityScapes(mode="val")
    elif validation_ds_name == "GTA5":
        val_dataset = gta5(mode="val")

    dataloader_val = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    # model
    model = BiSeNet(
        backbone=args.backbone,
        n_classes=n_classes,
        pretrain_model=args.pretrain_path,
        use_conv_last=args.use_conv_last,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: " + str(device))

    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # optimizer
    # build optimizer
    if args.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  # rmsprop
        logger.critical("not supported optimizer")
        return None

    if save_model_postfix != "":
        args.save_model_path = os.path.join(args.save_model_path, save_model_postfix)

    # train loop
    train(
        args,
        model,
        optimizer,
        dataloader_train,
        dataloader_val,
        training_name=save_model_postfix,
    )
    # final test
    val(args, model, dataloader_val)


if __name__ == "__main__":
    logger.info("tg:Starting MEGA training")
    # 2a
    try:
        logger.info("tg:Starting 2A Training")
        main("Cityscapes", "Cityscapes", save_model_postfix="2A")
    except Exception as e:
        logger.critical("tg:Error on 2A", exc_info=e)

    # 2b
    try:
        logger.info("tg:Starting 2B Training")
        main("GTA5", "GTA5", save_model_postfix="2B")
    except Exception as e:
        logger.critical("tg:Error on 2B", exc_info=e)
    # 2c.1
    try:
        logger.info("tg:Starting 2C1 Training")
        main("GTA5", "Cityscapes", save_model_postfix="2C1")
    except Exception as e:
        logger.critical("tg:Error on 2C1", exc_info=e)
    # 2c.2
    try:
        logger.info("tg:Starting 2C2 Training")
        main("GTA5", "Cityscapes", augmentation=True, save_model_postfix="2C2")
    except Exception as e:
        logger.critical("tg:Error on 2C2", exc_info=e)
# modified arguemnts: pretrain_path, num_epochs, batch_size, save_model_path
