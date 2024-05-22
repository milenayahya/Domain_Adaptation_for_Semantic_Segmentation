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
from model.discriminator import FCDiscriminator
from .options.train_ada_options import parse_args, TrainADAOptions, TrainOptions

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

            # visualzizing an image and logging
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


"""
Assigning source_label = 0 and target_label = 1: The idea is to train the discriminator to classify predictions 
as either coming from the source domain (assigned label 0) or the target domain (assigned label 1). 

During training, the aim is to minimize the ability of the discriminator to distinguish between the domains.
This is achieved by minimizing the binary cross-entropy loss between the discriminator's predictions 
and the assigned labels (source_label or target_label). 
As the training progresses, the discriminator becomes less effective at distinguishing between the domains, 
indicating that the features learned by the model become domain-invariant, which is our goal.
"""


def train(
    args: "TrainADAOptions",
    model: "torch.nn.Module",
    modelD: "torch.nn.Module",
    optimizer: "torch.optim.Optimizer",
    optimizerD: "torch.optim.Optimizer",
    dataloader_source: "DataLoader",
    dataloader_target: "DataLoader",
    dataloader_val: "DataLoader",
    validation_dataset_name: "DatasetName",
    training_name: Optional[str] = None,
    writer: Optional["SummaryWriter"] = None,
):
    if writer is None:
        writer = SummaryWriter(comment=f"")

    scaler = amp.GradScaler()

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)

    # bce_loss is minimized to minimize the ability of the discrimnator to distiguish between domains
    bce_loss = torch.nn.BCEWithLogitsLoss()

    precision, miou = 0, 0
    max_miou = 0
    step = 0

    # the labels which the discriminator will assign
    source_label = 0
    target_label = 1

    start_epoch = 1
    checkpoint_filename = os.path.join(args.save_model_path, "latest.tar")
    runs_execution_time = []
    if args.resume and os.path.exists(checkpoint_filename):
        checkpoint: dict = torch.load(checkpoint_filename)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        model_d_state_dict = checkpoint.get("state_dict_d", None)
        if model_d_state_dict is not None:
            modelD.load_state_dict(model_d_state_dict)
        optimizer_d_state_dict = checkpoint.get("optimizer_d", None)
        if optimizer_d_state_dict is not None:
            optimizerD.load_state_dict(optimizer_d_state_dict)
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
        lr_discriminator = poly_lr_scheduler(
            optimizerD, args.learning_rate_D, iter=(epoch - 1), max_iter=args.num_epochs
        )

        model.train()
        modelD.train()

        tq = tqdm(total=len(dataloader_target) * args.batch_size, unit="img")
        writer.add_scalar(f"{tensorboard_base_name}/learning_rate", lr, epoch)
        writer.add_scalar(
            f"{tensorboard_base_name}/learning_rate_discriminator",
            lr_discriminator,
            epoch,
        )
        tq.set_description(f"{epoch=}, {lr=}, {lr_discriminator}")

        loss_record = []
        loss_source_record = []
        loss_target_record = []

        for i, ((data, label), (data_target, _)) in zip(
            dataloader_source, dataloader_target
        ):

            data = data.cuda()
            label = label.long().cuda()
            data_target = data_target().cuda()

            optimizer.zero_grad()
            optimizerD.zero_grad()

            # Train generator with source data:
            # Discriminator model is frozen, params are not updated during back-propagation, no gradients computed
            for param in modelD.parameters():
                param.requires_grad = False

            with amp.autocast():
                output, out16, out32 = model(data)
                loss1 = loss_func(output, label.squeeze(1))
                loss2 = loss_func(out16, label.squeeze(1))
                loss3 = loss_func(out32, label.squeeze(1))
                loss = loss1 + loss2 + loss3

            scaler.scale(loss).backward()

            with amp.autocast():
                output_target, out16_target, out32_target = model(data_target)
                D_out1 = modelD(torch.softmax(output_target, dim=1))  # peppe-sc
                # D_out1 = modelD(F.softmax(output_target,dim=1)) #Alessio

                loss_adv_target1 = bce_loss(
                    D_out1,
                    torch.FloatTensor(D_out1.data.size()).fill_(source_label).cuda(),
                )
                loss_adv_target = args.lambda_d1 * loss_adv_target1

                ## in Alessio's repo loss is just loss1+loss2+loss3
                loss = loss + loss_adv_target  # peppe-sc

            scaler.scale(loss_adv_target).backward()

            # Now we train the Discriminator
            for param in modelD.parameters():
                param.requires_grad = True

            # detach tensors to prevent gradients from flowing back into the main model during the subsequent backward pass
            output = output.detach()
            out16 = out16.detach()
            out32 = out32.detach()

            # forward pass: apply softmax to the detached outputs and pass them through the Discriminator to obtain predictions
            with amp.autocast():
                D_out1 = modelD(torch.softmax(output, dim=1))
                # loss calculation
                loss_d_source1 = bce_loss(
                    D_out1,
                    torch.FloatTensor(D_out1.data.size()).fill_(source_label).cuda(),
                )

            # backward propagation
            scaler.scale(loss_d_source1).backward()

            # same for target domain:
            output_target = output_target.detach()
            out16_target = out16_target.detach()
            out32_target = out32_target.detach()

            with amp.autocast():
                D_out1 = modelD(torch.softmax(output_target, dim=1))
                loss_d_target1 = bce_loss(
                    D_out1,
                    torch.FloatTensor(D_out1.data.size()).fill_(target_label).cuda(),
                )

            scaler.scale(loss_d_target1).backward()

            scaler.step(optimizerD)
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

            # this loss is the loss of the generator for the source and target
            loss_record.append(loss.item())
            # discriminator loss for source
            loss_source_record.append(loss_d_source1.item())
            # discriminator loss for target
            loss_target_record.append(loss_d_target1.item())

        tq.close()
        loss_train_mean = np.mean(loss_record)
        loss_discr_source_mean = np.mean(loss_source_record)
        loss_discr_target_mean = np.mean(loss_target_record)

        writer.add_scalar(
            f"{tensorboard_base_name}/loss_epoch_train",
            float(loss_train_mean),
            epoch,
            display_name="Mean Loss per Epoch",
        )
        writer.add_scalar(
            f"{tensorboard_base_name}/loss_epoch_discr_source",
            float(loss_discr_source_mean),
            epoch,
            display_name="Mean Loss per Epoch (Discrimator Source)",
        )
        writer.add_scalar(
            f"{tensorboard_base_name}/loss_epoch_discr_target",
            float(loss_discr_target_mean),
            epoch,
            display_name="Mean Loss per Epoch (Discrimator Target)",
        )

        ts_duration: timedelta = datetime.today() - ts_start
        logger.info(
            f"loss for train @ {epoch=}: {loss_train_mean=} | {loss_discr_source_mean=} | {loss_discr_target_mean=} after {ts_duration.seconds} seconds"
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
                    "state_dict_d": modelD.state_dict(),
                    "max_miou": max_miou,
                    "precision": precision,
                    "optimizer": optimizer.state_dict(),
                    "optimizer_d": optimizerD.state_dict(),
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
                        "state_dict_d": modelD.state_dict(),
                        "max_miou": max_miou,
                        "precision": precision,
                        "optimizer": optimizer.state_dict(),
                        "optimizer_d": optimizerD.state_dict(),
                    },
                    args.save_model_path,
                    True,
                )
    logger.info(
        f"tg:Finished training of {training_name} after an average of {sum(runs_execution_time)/len(runs_execution_time)} seconds per epoch. Best was reached @ {best_epoch} epoch"
    )


def main(
    source_ds_name: DatasetName,
    target_ds_name: DatasetName,
    validation_ds_name: DatasetName,
    *,
    augmentation: bool = False,
    save_model_postfix: str = "",
    args: Optional["TrainADAOptions"] = None,
    writer: Optional["SummaryWriter"] = None,
) -> tuple[float, float]:
    if args is None:
        args = parse_args()

    # dataset
    n_classes = args.num_classes

    # Source dataset
    if source_ds_name == "Cityscapes":
        source_dataset = Cityscapes(
            mode="train",
            transforms=OurCompose([OurResize(CITYSCAPES_CROP_SIZE), OurToTensor()]),
        )
    elif source_ds_name == "GTA5":
        if augmentation:
            transformations = OurCompose(
                [
                    OurToTensor(),
                    OurRandomCrop(GTA5_CROP_SIZE),
                    OurGeometricAugmentationTransformations(),
                    OurColorJitterTransformation(),
                ]
            )
        else:
            transformations = OurCompose([OurResize(GTA5_CROP_SIZE), OurToTensor()])

        source_dataset = GTA5("train", transforms=transformations)
    else:
        raise ValueError("Dataset non valido")

    dataloader_source = DataLoader(
        source_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
    )

    # Target dataset
    if target_ds_name == "Cityscapes":
        target_dataset = Cityscapes(
            mode="train",
            transforms=OurCompose([OurResize(CITYSCAPES_CROP_SIZE), OurToTensor()]),
        )
    elif target_ds_name == "GTA5":
        if augmentation:
            transformations = OurCompose(
                [
                    OurToTensor(),
                    OurRandomCrop(GTA5_CROP_SIZE),
                    OurGeometricAugmentationTransformations(),
                    OurColorJitterTransformation(),
                ]
            )
        else:
            transformations = OurCompose([OurResize(GTA5_CROP_SIZE), OurToTensor()])

        target_dataset = GTA5("train", transforms=transformations)
    else:
        raise ValueError("Dataset non valido")

    dataloader_target = DataLoader(
        target_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
    )

    # Validation dataset
    if validation_ds_name == "Cityscapes":
        val_dataset = Cityscapes(mode="val", transforms=OurToTensor())
    elif validation_ds_name == "GTA5":
        val_dataset = GTA5(mode="val", transforms=OurToTensor())
    else:
        raise ValueError("Dataset non valido")

    dataloader_val = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    # model (Generator)
    model = BiSeNet(
        backbone=args.backbone,
        n_classes=n_classes,
        pretrain_model=str(args.pretrain_path),
        use_conv_last=args.use_conv_last,
    )

    # Discriminator model
    model_D = FCDiscriminator(num_classes=args.num_classes)
    # create more instances to implement multi-level discriminator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug("Device: " + str(device))

    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()
        model_D = torch.nn.DataParallel(model_D).cuda()

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

    optimizer_D = torch.optim.Adam(
        model_D.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99)
    )

    if save_model_postfix != "":
        args.save_model_path = Path(
            os.path.join(args.save_model_path, save_model_postfix)
        )

    # train loop
    if args.mode == "train":
        train(
            args,
            model,
            model_D,
            optimizer,
            optimizer_D,
            dataloader_source,
            dataloader_target,
            dataloader_val,
            training_name=save_model_postfix,
            writer=writer,
            validation_dataset_name=validation_ds_name,
        )

    # final test
    checkpoint_filename = os.path.join(args.save_model_path, "best.tar" if args.use_best else "latest.tar")
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
    logger.info(f"Loaded {args.use_best and "best" or "latest"} checkpoint that was at epoch n. {start_epoch}")
    
    precision, max_miou = val(
        args,
        model,
        dataloader_val,
        writer=writer,
        name=save_model_postfix,
        visualize_images=True,
        epoch=args.num_epochs + 10,  # above anything, final validation
        dataset_name=validation_ds_name,
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


if __name__ == "__main__":

    logger.info("tg:Starting MEGA ADA")
    try:
        logger.info("tg:Starting task 3: ADA, GTA5 -> Cityscapes")
        writer = SummaryWriter(comment="task_3")
        precision_3, miou_3 = main(
            "GTA5",
            "Cityscapes",
            "Cityscapes",
            save_model_postfix="3/normal",
            writer=writer,
        )
        logger.info(f"tg:3 Results: Precision={precision_3} Mean IoU={miou_3}")
    except Exception as e:
        logger.critical("tg:Error on 3GCC", exc_info=e)

    try:
        logger.info("tg:Starting task 3: ADA, GTA5+aug -> Cityscapes")
        writer = SummaryWriter(comment="task_3_aug")
        precision_3_aug, miou_3_aug = main(
            "GTA5",
            "Cityscapes",
            "Cityscapes",
            augmentation=True,
            save_model_postfix="3/aug",
            writer=writer,
        )
        logger.info(
            f"tg:3_aug Results: Precision={precision_3_aug} Mean IoU={miou_3_aug}"
        )
    except Exception as e:
        logger.critical("tg:Error on 3_aug", exc_info=e)


# Doubts:

# use F.softmax or FN.softmax to compute output of models ?
# -> Using torch.softmax

# Training Generator loss: do we add the discriminator loss (peppe-sc) or not (Alessio) ?

# should we implement the multilevel discriminator (D1, D2, D3) ? -> multiple instances of discriminator,
# each uses outputs from a different level of the NN ?

# we could hypertune lambda, as for the moment we only use lambda_d1
