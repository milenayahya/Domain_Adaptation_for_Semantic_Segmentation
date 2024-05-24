#!/usr/bin/python
# -*- encoding: utf-8 -*-
try:
    from Domain_Adaptation_for_Semantic_Segmentation.Datasets import CITYSCAPES_CROP_SIZE, GTA5_CROP_SIZE, PROJECT_BASE_PATH  # type: ignore
except ImportError:
    from Datasets import CITYSCAPES_CROP_SIZE, GTA5_CROP_SIZE, PROJECT_BASE_PATH  # type: ignore
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Optional

from Datasets.transformations import *
from model.model_stages import BiSeNet
import torch
from torch.utils.data import DataLoader
import numpy as np
from tensorboardX import SummaryWriter
import torch.cuda.amp as amp
from tqdm import tqdm
import os
from Datasets.cityscapes_torch import Cityscapes
from Datasets.gta5 import GTA5
from .options.train_fda_options import parse_args, TrainFDAOptions, TrainOptions
from model.fda_utils import FDAEntropyLoss, FDA_source_to_target

from utils import (
    fast_compute_global_accuracy,
    reverse_one_hot,
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
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_BASE_PATH, "logs/debug.log")),
        logging.StreamHandler(),
        tg_handler,
    ],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


DatasetName = Literal["Cityscapes", "GTA5"]

# MAYBE SHOULD USE IMAGENET MEAN??? INVESTIGATE THIS
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
IMG_MEAN = torch.reshape(torch.from_numpy(IMG_MEAN), (1, 3, 1, 1))


def val(
    args: "TrainOptions",
    model: "torch.nn.Module",
    dataloader: "DataLoader",
    visualize_images: bool = False,
    writer: Optional["SummaryWriter"] = None,
    name: Optional[str] = None,
    epoch: Optional[int] = None,
    dataset_name: Optional[DatasetName] = "Cityscapes",
) -> tuple:

    visualize_img_idx = 0  # random.randrange(0, len(dataloader))

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

            # visualizing images and writing logs
            if (
                writer is not None
                and name is not None
                and epoch is not None
                and dataset_name is not None
                and visualize_images
                and i == visualize_img_idx
            ):
                logger.info(
                    f"Visualizing Prediction of {name}, picked image at index {visualize_img_idx}"
                )
                if dataset_name == "Cityscapes":
                    vp, vl = Cityscapes.visualize_prediction(predict, label)
                elif dataset_name == "GTA5":
                    vp, vl = GTA5.visualize_prediction(predict, label)
                writer.add_image(
                    f"{name}/prediction",
                    np.array(vp),
                    global_step=epoch,
                    dataformats="HWC",
                )
                writer.add_image(
                    f"{name}/ground_truth",
                    np.array(vl),
                    global_step=epoch,
                    dataformats="HWC",
                )
                writer.add_image(
                    f"{name}/source",
                    np.array(data[0].cpu()),
                    global_step=epoch,
                    dataformats="CHW",
                )
                writer.flush()

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
        if writer is not None and epoch is not None:
            writer.add_scalar(
                f"{name}/precision",
                precision,
                epoch,
                display_name="Precision",
            )
            writer.add_scalar(f"{name}/miou", miou, epoch, display_name="Mean IoU")
        return precision, miou


def train(
    args: "TrainFDAOptions",
    model: "torch.nn.Module",
    optimizer: "torch.optim.Optimizer",
    dataloader_train: "DataLoader",
    dataloader_target: "DataLoader",
    dataloader_val: "DataLoader",
    validation_dataset_name: "DatasetName",
    training_name: Optional[str] = None,
    writer: Optional["SummaryWriter"] = None,
):
    if writer is None:
        writer = SummaryWriter(comment=f"")

    scaler = amp.GradScaler()

    ce_loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)

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
        precision = checkpoint.get("precision", 0)
        logger.info(
            f"Loaded latest checkpoint that was at epoch n. {start_epoch}, {precision=} {max_miou=}"
        )

        if start_epoch >= args.num_epochs + 1:
            logger.warning("Since checkpoint is already complete, skipping training")
            return

    tensorboard_base_name = (
        f"{training_name}"
        if (training_name is None or ("/" not in training_name))
        else training_name.split("/")[0]
    )
    best_epoch = 0
    for epoch in range(start_epoch, args.num_epochs + 1):
        ts_start = datetime.today()
        lr = poly_lr_scheduler(
            optimizer, args.learning_rate, iter=(epoch - 1), max_iter=args.num_epochs
        )  # epoch-1 because we are now starting from 1
        model.train()
        tq = tqdm(total=len(dataloader_train) * args.batch_size, unit="img")
        writer.add_scalar(f"{tensorboard_base_name}/learning_rate", lr, epoch)
        tq.set_description("epoch %d, lr %f" % (epoch, lr))
        loss_record = []
        steps_to_do = min(len(dataloader_train), len(dataloader_target))
        random_step_to_visualize = random.randrange(0, steps_to_do) + step
        for it_train, it_target in zip(dataloader_train, dataloader_target):
            data, label = it_train

            data_target, _ = it_target  # Used for entropy loss

            data = data.cuda()
            data_copy = data.clone()
            label = label.long().cuda()

            optimizer.zero_grad()

            # Image preprocessing before feeding to the model
            if mean_img.shape[-1] < 2:
                B, C, H, W = data.shape
                mean_img = IMG_MEAN.repeat(B, 1, H, W)  # type: ignore

            # 1. source to target, target to target
            src_in_trg = FDA_source_to_target(data, data_target, L=args.fda_beta)
            trg_in_trg = data_target

            # 2. subtract mean
            data = src_in_trg.clone() - mean_img
            if step == random_step_to_visualize:
                # Extract single image from batch of both data_copy and data
                data_copy_vis = data_copy[:, :, 0:1, :]
                data_vis = data[:, :, 0:1, :]
                # Now log it via writer
                writer.add_image(
                    f"{tensorboard_base_name}/fda_visualization_pre",
                    np.array(data_copy_vis),
                    global_step=epoch,
                    dataformats="CHW",
                )
                writer.add_image(
                    f"{tensorboard_base_name}/fda_visualization_post",
                    np.array(data_vis),
                    global_step=epoch,
                    dataformats="CHW",
                )
                writer.flush()

            with amp.autocast():
                output, out16, out32 = model(data)
                loss1 = ce_loss_func(output, label.squeeze(1))
                loss2 = ce_loss_func(out16, label.squeeze(1))
                loss3 = ce_loss_func(out32, label.squeeze(1))
                loss = loss1 + loss2 + loss3

                # CALCULATE ENTROPY LOSS HERE
                if epoch > args.switch_to_entropy_after_epoch:
                    # 2. subtract mean also here, but only when needed..
                    # is it expensive? not sure but it costs nothing to try
                    data_target = trg_in_trg.clone() - mean_img
                    target_output, _, _ = model(data_target)
                    target_loss = FDAEntropyLoss(target_output, args.eta)
                    loss = loss + args.ent_loss_scaling * target_loss
                    writer.add_scalar(
                        f"{tensorboard_base_name}/ent_loss_step",
                        target_loss,
                        step,
                        display_name="Entropy Loss per Step",
                    )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tq.update(args.batch_size)
            tq.set_postfix(loss="%.6f" % loss)
            step += 1
            writer.add_scalar(
                f"{tensorboard_base_name}/loss_step",
                loss,
                step,
                display_name="Loss per Step",
            )
            loss_record.append(loss.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar(
            f"{tensorboard_base_name}/loss_epoch",
            float(loss_train_mean),
            epoch,
            display_name="Mean Loss per Epoch",
        )
        ts_duration: timedelta = datetime.today() - ts_start
        logger.info(
            f"loss for train @ {epoch=}: {loss_train_mean} after {ts_duration.seconds} seconds"
        )
        runs_execution_time.append(ts_duration.seconds)
        writer.add_scalar(
            f"{tensorboard_base_name}/training_duration",
            ts_duration.seconds,
            epoch,
            display_name="Training Duration",
            summary_description="seconds",
        )
        if epoch % args.checkpoint_step == 0:
            logger.info(
                f"Saving latest checkpoint @ {epoch=}, with max_miou (of latest val) = {max_miou}"
            )
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "max_miou": max_miou,
                    "precision": precision,
                    "optimizer": optimizer.state_dict(),
                    "name": training_name,
                },
                args.save_model_path,
                False,
            )

        if epoch % args.validation_step == 0:
            precision, miou = val(
                args,
                model,
                dataloader_val,
                writer=writer,
                name=tensorboard_base_name,
                epoch=epoch,
                visualize_images=True,
                dataset_name=validation_dataset_name,
            )
            if miou > max_miou:
                max_miou = miou
                best_epoch = epoch
                logger.info(f"Saving best checkpoint @ {epoch=}, with {max_miou=}")
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "max_miou": max_miou,
                        "precision": precision,
                        "optimizer": optimizer.state_dict(),
                        "name": training_name,
                    },
                    args.save_model_path,
                    True,
                )
    logger.info(
        f"tg:Finished training of {training_name} after an average of {sum(runs_execution_time)/len(runs_execution_time)} seconds per epoch. Best was reached @ {best_epoch} epoch"
    )


def main(
    *,
    save_model_postfix: str = "",
    args: Optional[TrainFDAOptions] = None,
    writer: Optional["SummaryWriter"] = None,
) -> tuple[float, float]:
    if args is None:
        args = parse_args()

    # dataset
    n_classes = args.num_classes

    train_dataset = GTA5(
        "train",
        transforms=OurCompose(
            [
                OurResize(GTA5_CROP_SIZE),
                OurToTensor(),
                OurRandomCrop(CITYSCAPES_CROP_SIZE),
            ]
        ),
    )

    target_dataset = Cityscapes(
        mode="train",
        transforms=OurCompose([OurResize(CITYSCAPES_CROP_SIZE), OurToTensor()]),
    )

    dataloader_train = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
    )
    dataloader_target = DataLoader(
        target_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
    )

    # Validation dataset
    val_dataset = Cityscapes(
        mode="val",
        transforms=OurCompose([OurResize(CITYSCAPES_CROP_SIZE), OurToTensor()]),
    )

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
        pretrain_model=str(args.pretrain_path),
        use_conv_last=args.use_conv_last,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug("Device: " + str(device))

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
        raise Exception("not supported optimizer")

    if save_model_postfix != "":
        args.save_model_path = Path(
            os.path.join(args.save_model_path, save_model_postfix)
        )

    # train loop
    if args.mode == "train":
        train(
            args,
            model,
            optimizer,
            dataloader_train,
            dataloader_target,
            dataloader_val,
            training_name=save_model_postfix,
            writer=writer,
            validation_dataset_name="Cityscapes",
        )

    # final test
    checkpoint_filename = os.path.join(
        args.save_model_path, "best.tar" if args.use_best else "latest.tar"
    )
    logger.info(
        f"Performing final evaluation with the {args.use_best and "best" or "latest"} model saved in the following checkpoint: {checkpoint_filename}"
    )
    # Load Best before final evaluation
    checkpoint: dict[str, Any]
    if not os.path.exists(checkpoint_filename):
        raise Exception(
            f"Trying to train on {args.use_best and "best" or "latest"} but it doesn't exitsts. Looking for {checkpoint_filename}"
        )

    checkpoint = torch.load(checkpoint_filename)
    start_epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["state_dict"])
    logger.info(
        f"Loaded {args.use_best and "best" or "latest"} checkpoint that was at epoch n. {start_epoch-1}"
    )

    precision, max_miou = val(
        args,
        model,
        dataloader_val,
        writer=writer,
        name=save_model_postfix,
        visualize_images=True,
        epoch=args.num_epochs + 10,  # above anything, final validation
        dataset_name="Cityscapes",
    )

    # Save again best model, but with updated precision and max_miou
    if (
        checkpoint_filename is not None
        and checkpoint is not None
        and os.path.exists(checkpoint_filename)
    ):
        checkpoint["precision"] = precision
        checkpoint["max_miou"] = max_miou
        if "name" not in checkpoint.keys():
            checkpoint["name"] = save_model_postfix
        save_checkpoint(checkpoint, args.save_model_path, True)

    return precision, max_miou
