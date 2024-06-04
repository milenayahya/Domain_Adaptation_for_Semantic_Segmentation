#!/usr/bin/python
# -*- encoding: utf-8 -*-
try:
    from Domain_Adaptation_for_Semantic_Segmentation.Datasets import CITYSCAPES_CROP_SIZE, GTA5_CROP_SIZE, PROJECT_BASE_PATH  # type: ignore
except ImportError:
    from Datasets import CITYSCAPES_CROP_SIZE, GTA5_CROP_SIZE, PROJECT_BASE_PATH  # type: ignore
import logging
from datetime import datetime, timedelta
from pathlib import Path
from pprint import pformat
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
from .options.train_options import parse_args, TrainOptions

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
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_BASE_PATH, "logs/debug.log")),
        logging.StreamHandler(),
        tg_handler,
    ],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


DatasetName = Literal["Cityscapes", "GTA5"]
Tasks = Literal["2A", "2B", "2C1", "2C2"]


def val(
    args: "TrainOptions",
    model: "torch.nn.Module",
    dataloader: "DataLoader",
    visualize_images: bool = False,
    writer: Optional["SummaryWriter"] = None,
    name: Optional[str] = None,
    epoch: Optional[int] = None,
    dataset_name: Optional[DatasetName] = None,
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
    args: "TrainOptions",
    model: "torch.nn.Module",
    optimizer: "torch.optim.Optimizer",
    dataloader_train: "DataLoader",
    dataloader_val: "DataLoader",
    validation_dataset_name: "DatasetName",
    training_name: Optional[str] = None,
    writer: Optional["SummaryWriter"] = None,
):
    if writer is None:
        writer = SummaryWriter(comment=f"")

    scaler = amp.GradScaler()

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    precision, miou = 0, 0
    max_miou = 0
    step = 0
    start_epoch = 1

    # continue training from last checkpoint
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
    training_ds_name: DatasetName,
    validation_ds_name: DatasetName,
    *,
    augmentation: bool = False,
    save_model_postfix: str = "",
    args: Optional[TrainOptions] = None,
    writer: Optional["SummaryWriter"] = None,
) -> tuple[float, float]:
    if args is None:
        args = parse_args()

    # dataset
    n_classes = args.num_classes

    if training_ds_name == "Cityscapes":
        train_dataset = Cityscapes(
            mode="train",
            transforms=OurCompose(
                [OurResize(CITYSCAPES_CROP_SIZE), OurToTensor(), OurNormalization()]
            ),
        )
    elif training_ds_name == "GTA5":
        if augmentation:
            transformations = OurCompose(
                [
                    OurToTensor(),
                    OurNormalization(),
                    OurRandomCrop(GTA5_CROP_SIZE),
                    OurGeometricAugmentationTransformations(),
                    OurColorJitterTransformation(),
                ]
            )
        else:
            transformations = OurCompose(
                [OurResize(GTA5_CROP_SIZE), OurToTensor(), OurNormalization()]
            )

        train_dataset = GTA5("train", transforms=transformations)
    else:
        raise ValueError("Dataset non valido")

    dataloader_train = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
    )

    # Validation dataset
    if validation_ds_name == "Cityscapes":
        val_dataset = Cityscapes(
            mode="val",
            transforms=OurCompose(
                [OurResize(CITYSCAPES_CROP_SIZE), OurToTensor(), OurNormalization()]
            ),
        )
    elif validation_ds_name == "GTA5":
        val_dataset = GTA5(
            mode="val",
            transforms=OurCompose(
                [OurResize(GTA5_CROP_SIZE), OurToTensor(), OurNormalization()]
            ),
        )
    else:
        raise ValueError("Dataset non valido")

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
        logger.critical("not supported optimizer")
        return (-1, -1)

    if "2C1" in save_model_postfix:
        # IF task 2c.1 -> Re use model of 2b and valuate only on Cityscape directly
        args.mode = "validate_with_best"
        save_model_postfix = save_model_postfix.replace("2C1", "2B")

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
            dataloader_val,
            training_name=save_model_postfix,
            writer=writer,
            validation_dataset_name=validation_ds_name,
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
        name=save_model_postfix.split("/")[0],
        visualize_images=True,
        epoch=args.num_epochs + 10,  # above anything, final validation
        dataset_name=validation_ds_name,
    )

    # Save again best model, but with updated precision and max_miou
    if (
        checkpoint_filename is not None
        and checkpoint is not None
        and os.path.exists(checkpoint_filename)
        and args.mode == "train"
    ):
        checkpoint["precision"] = precision
        checkpoint["max_miou"] = max_miou
        if "name" not in checkpoint.keys():
            checkpoint["name"] = save_model_postfix
        save_checkpoint(checkpoint, args.save_model_path, True)

    return precision, max_miou


def run2A(
    args: Optional[TrainOptions] = None,
    name: str = "2A",
    writer: Optional["SummaryWriter"] = None,
):
    if "2A" not in name:
        name = f"2A/{name}"
    precision, miou = None, None
    try:
        logger.info(f"tg:Starting {name} Training")
        precision, miou = main(
            "Cityscapes",
            "Cityscapes",
            save_model_postfix=name,
            args=args,
            writer=writer,
        )
        logger.info(f"tg:{name} Results: Precision={precision} Mean IoU={miou}")
    except Exception as e:
        logger.critical(f"tg:Error on {name}", exc_info=e)
    return {"precision": precision, "miou": miou}


def run2B(
    args: Optional[TrainOptions] = None,
    name: str = "2B",
    writer: Optional["SummaryWriter"] = None,
):
    if "2B" not in name:
        name = f"2B/{name}"
    precision, miou = None, None
    try:
        logger.info(f"tg:Starting {name} Training")
        precision, miou = main(
            "GTA5", "GTA5", save_model_postfix=name, args=args, writer=writer
        )
        logger.info(f"tg:{name} Results: Precision={precision} Mean IoU={miou}")
    except Exception as e:
        logger.critical(f"tg:Error on {name}", exc_info=e)
    return {"precision": precision, "miou": miou}


def run2C1(
    args: Optional[TrainOptions] = None,
    name: str = "2C1",
    writer: Optional["SummaryWriter"] = None,
):
    if "2C1" not in name:
        name = f"2C1/{name}"
    precision, miou = None, None
    try:
        logger.info(f"tg:Starting {name} valuation")
        precision, miou = main(
            "GTA5", "Cityscapes", save_model_postfix=name, args=args, writer=writer
        )
        logger.info(f"tg:{name} Results: Precision={precision} Mean IoU={miou}")
    except Exception as e:
        logger.critical(f"tg:Error on {name}", exc_info=e)
    return {"precision": precision, "miou": miou}


def run2C2(
    args: Optional[TrainOptions] = None,
    name: str = "2C2",
    writer: Optional["SummaryWriter"] = None,
):
    if "2C2" not in name:
        name = f"2C2/{name}"
    precision, miou = None, None
    try:
        logger.info(f"tg:Starting {name} Training")
        precision, miou = main(
            "GTA5",
            "Cityscapes",
            augmentation=True,
            save_model_postfix=name,
            args=args,
            writer=writer,
        )
        logger.info(f"tg:{name} Results: Precision={precision} Mean IoU={miou}")
    except Exception as e:
        logger.critical(f"tg:Error on {name}", exc_info=e)
    return {"precision": precision, "miou": miou}


def run(tasks: Optional[dict[Tasks, TrainOptions]] = None):
    if tasks is None or len(tasks) == 0:
        # run them all with parsed arguments
        tasks = {
            "2A": TrainOptions().from_dict({"batch_size": 4, "optimizer": "sgd"}),
            "2B": TrainOptions().from_dict({"batch_size": 4, "optimizer": "sgd"}),
            "2C1": TrainOptions().from_dict({"batch_size": 6, "optimizer": "sgd"}),
            "2C2": TrainOptions().from_dict({"batch_size": 6, "optimizer": "sgd"}),
        }
    results: dict[str, Any] = dict()
    if "2A" in tasks:
        args = tasks["2A"]
        writer = SummaryWriter(comment="BEST_EVAL_2A_VR-SGD-6")
        res = run2A(args=args, name="2A/VR-SGD-6", writer=writer)
        results["2A"] = res
    # # 2b
    if "2B" in tasks:
        args = tasks["2B"]
        writer = SummaryWriter(comment="BEST_EVAL_2B_VR-SGD-6")
        res = run2B(args=args, name="2B/VR-SGD-6", writer=writer)
        results["2B"] = res

    # 2c.1
    if "2C1" in tasks:
        args = tasks["2C1"]
        writer = SummaryWriter(comment="BEST_EVAL_2C1_SGD-6")
        res = run2C1(args=args, name="2C1/SGD-6", writer=writer)
        results["2C1"] = res

    # 2c.2
    if "2C2" in tasks:
        args = tasks["2C2"]
        writer = SummaryWriter(comment="BEST_EVAL_2C2_SGD-6")
        res = run2C2(args=args, name="2C2/SGD-6", writer=writer)
        results["2C2"] = res

    return results

def grid_search():
    # Batch size: 4 or 6 or 8
    # Optimizer: SGD, ADAM
    GRID: dict[str, dict] = {
        "ADAM-4": {"batch_size": 4, "optimizer": "adam"},
        "SGD-4": {"batch_size": 4, "optimizer": "sgd"},
        "RMS-4": {"batch_size": 4, "optimizer": "rmsprop"},
        "ADAM-6": {"batch_size": 6, "optimizer": "adam"},
        "SGD-6": {"batch_size": 6, "optimizer": "sgd"},
        "RMS-6": {"batch_size": 6, "optimizer": "rmsprop"},
        "ADAM-2": {"batch_size": 2, "optimizer": "adam"},
        "SGD-2": {"batch_size": 2, "optimizer": "sgd"},
        "RMS-2": {"batch_size": 2, "optimizer": "rmsprop"},
        "ADAM-8": {"batch_size": 8, "optimizer": "adam"},
        "SGD-8": {"batch_size": 8, "optimizer": "sgd"},
        "RMS-8": {"batch_size": 8, "optimizer": "rmsprop"},
    }
    RES_2A: dict[str, dict[str, Optional[float]]] = dict()
    RES_2B: dict[str, dict[str, Optional[float]]] = dict()
    logger.info(f"tg: Starting GRID SEARCH: {list(GRID.keys())}")
    for name, args in GRID.items():
        writer = SummaryWriter(comment=f"_BIGGEST_GRID_SEARCH_OVER_NIGHT_{name}")
        RES_2A[name] = run2A(
            args=TrainOptions().from_dict(args), name=f"2A/{name}", writer=writer
        )
        RES_2B[name] = run2B(
            args=TrainOptions().from_dict(args), name=f"2B/{name}", writer=writer
        )
    logger.info(f"tg: Finished GRID SEARCH:\n{pformat(RES_2A, indent=2, underscore_numbers=True)}\n{pformat(RES_2B, indent=2, underscore_numbers=True)}")


if __name__ == "__main__":
    tasks_to_run: list[Tasks] = ["2C1", "2C2"]
    logger.info(f"tg:Starting the following TASKS: {tasks_to_run} on SGD-6")
    tasks: dict[Tasks, TrainOptions] = {t: parse_args() for t in tasks_to_run}
    res = run(tasks=tasks)
    logger.info(f"tg:Finished the following TASKS: {tasks_to_run}, results: {pformat(res, indent=2, underscore_numbers=True)}")
    # grid_search()
