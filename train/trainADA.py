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
from model.discriminator import FCDiscriminator
import torchvision.transforms.functional as F
import torch.nn.functional as FN
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
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "logs/debug.log")),
        logging.StreamHandler(),
        tg_handler,
    ],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


DatasetName = Literal["Cityscapes", "GTA5"]


def val(args, model, dataloader) -> tuple[float, float]:
    with torch.no_grad():
        model.eval()
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        pbar = tqdm(total=len(dataloader), desc="Evaluation", unit="img")  #progress bar
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


'''
Assigning source_label = 0 and target_label = 1: The idea is to train the discriminator to classify predictions 
as either coming from the source domain (assigned label 0) or the target domain (assigned label 1). 

During training, the aim is to minimize the ability of the discriminator to distinguish between the domains.
This is achieved by minimizing the binary cross-entropy loss between the discriminator's predictions 
and the assigned labels (source_label or target_label). 
As the training progresses, the discriminator becomes less effective at distinguishing between the domains, 
indicating that the features learned by the model become domain-invariant, which is our goal.
'''

def train(
    args, model, modelD, optimizer, optimizerD, dataloader_source, dataloader_target, dataloader_val, training_name: str = None
):
    writer = SummaryWriter(comment=f"{training_name}{args.optimizer}")

    scaler = amp.GradScaler()

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)

    #bce_loss is minimized to minimize the ability of the discrimnator to distiguish between domains
    bce_loss = torch.nn.BCEWithLogitsLoss()

    precision, miou = 0, 0
    max_miou = 0
    step = 0

    #the labels which the discriminator will assign
    source_label= 0
    target_label= 1

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

    tensorboard_base_name = "epoch/"
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
        writer.add_scalar(f"{tensorboard_base_name}/learning_rate_discriminator", lr_discriminator, epoch)
        tq.set_description("epoch %d, lr %f, lr_discriminator %f" (epoch, lr, lr_discriminator))


        loss_record = []
        loss_source_record= []
        loss_target_record= []

        for i, ((data, label),(data_target,_)) in zip(dataloader_source, dataloader_target):

            data = data.cuda()
            label = label.long().cuda()
            data_target= data_target().cuda()

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
                D_out1 = modelD(FN.softmax(output_target,dim=1)) #peppe-sc
                #D_out1 = modelD(F.softmax(output_target,dim=1)) #Alessio

                loss_adv_target1= bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(source_label).cuda())
                loss_adv_target = args.lambda_d1*loss_adv_target1
                
                ## in Alessio's repo loss is just loss1+loss2+loss3
                loss = loss + loss_adv_target #peppe-sc

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
                D_out1= modelD(FN.softmax(output, dim=1))
                #loss calculation
                loss_d_source1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill(source_label).cuda())
            
            #backward propagation
            scaler.scale(loss_d_source1).backward()

            # same for target domain:
            output_target= output_target.detach()
            out16_target= out16_target.detach()
            out32_target= out32_target.detach()

            with amp.autocast():
                D_out1 = modelD(FN.softmax(output_target,dim=1))
                loss_d_target1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill(target_label).cuda())

            scaler.scale(loss_d_target1).backward()

            scaler.step(optimizerD)
            scaler.step(optimizer)
            scaler.update()

            tq.update(args.batch_size)
            tq.set_postfix(loss="%.6f" % loss)
            step += 1
            writer.add_scalar("step/loss", loss, step)

            loss_record.append(loss.item()) # this loss is the loss of the generator for the source and target
            loss_source_record.append(loss_d_source1.item()) # discriminator loss for source
            loss_target_record.append(loss_d_target1.item()) # discriminator loss for target

        tq.close()
        loss_train_mean = np.mean(loss_record)
        loss_discr_source_mean = np.mean(loss_source_record)
        loss_discr_target_mean = np.mean(loss_target_record)

        writer.add_scalar(
            f"{tensorboard_base_name}/loss_epoch_train", float(loss_train_mean), epoch
        )
        writer.add_scalar(
            f"{tensorboard_base_name}/loss_epoch_discr_source", float(loss_discr_source_mean), epoch
        )
        writer.add_scalar(
            f"{tensorboard_base_name}/loss_epoch_discr_target", float(loss_discr_target_mean), epoch
        )

        ts_duration: timedelta = datetime.today() - ts_start
        logger.info(
            f"loss for train @ {epoch=}: {loss_train_mean} after {ts_duration.seconds} seconds"
        )
        logger.info(
            f"loss for discriminant on source domain @ {epoch=}: {loss_discr_source_mean} after {ts_duration.seconds} seconds"
        )
        logger.info(
            f"loss for discriminant on target domain @ {epoch=}: {loss_discr_target_mean} after {ts_duration.seconds} seconds"
        )
        runs_execution_time.append(ts_duration.seconds)
        writer.add_scalar(
            f"{tensorboard_base_name}/training_duration",
            ts_duration.seconds,
            epoch,
            display_name="Training Duration",
            summary_description="seconds",
        )


        ## check from 272 to 318 if anything needs to be changed
        if epoch % args.checkpoint_step == 0:
            logger.info(
                f"Saving latest checkpoint @ {epoch=}, with max_miou (of latest val) = {max_miou}"
            )
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
            writer.add_scalar(
                f"{tensorboard_base_name}/precision",
                precision,
                epoch,
                display_name="Precision",
            )
            writer.add_scalar(
                f"{tensorboard_base_name}/miou", miou, epoch, display_name="Mean IoU"
            )
            if miou > max_miou:
                max_miou = miou
                best_epoch = epoch
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
    logger.info(
        f"tg:Finished training of {training_name} after {sum(runs_execution_time)/len(runs_execution_time)} seconds. Best was reached @ {best_epoch} epoch"
    )


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
        choices=["train", "val_with_best"],
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
        "--use_best",
        action="store_true",
        help="Use best model for final validation",
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
        "--batch_size", type=int, default=8, help="Number of images in each batch"
    )
    parse.add_argument(
        "--learning_rate", type=float, default=0.01, help="learning rate used for train"
    )
    parse.add_argument('--learning_rate_D',
                        type=float,
                        default=0.0001, #default=0.0002,
                        help='learning rate used for train')
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
        default="sgd",
        help="optimizer, support rmsprop, sgd, adam",
    )
    parse.add_argument("--loss", type=str, default="crossentropy", help="loss function")
    parse.add_argument('--lambda_d1',
                       type=float,
                       default=0.001,
                       help='lambda for adversarial loss')
    parse.add_argument('--lambda_d2',
                       type=float,
                       default=0.0002,
                       help='lambda for adversarial loss')
    parse.add_argument('--lambda_d3',
                       type=float,
                       default=0.0002,
                       help='lambda for adversarial loss')
    return parse.parse_args()


def main(
    source_ds_name: DatasetName,
    target_ds_name: DatasetName,
    validation_ds_name: DatasetName,
    *,
    augmentation: bool = False,
    save_model_postfix: str = "",
) -> tuple[float, float]:
    args = parse_args()

    # dataset
    n_classes = args.num_classes

    mode = args.mode

    # Source dataset
    if source_ds_name == "Cityscapes":
        source_dataset = CityScapes("train")
    elif source_ds_name == "GTA5":
        source_dataset = gta5("train", aug=augmentation)
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
        target_dataset = CityScapes(mode="train")
    elif target_ds_name == "GTA5":
        target_dataset = gta5(mode="train")
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
        val_dataset = CityScapes(mode="val")
    elif validation_ds_name == "GTA5":
        val_dataset = gta5(mode="val")
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
        pretrain_model=args.pretrain_path,
        use_conv_last=args.use_conv_last,
    )

    # Discriminator model
    model_D = FCDiscriminator(num_classes=args.num_classses)
    # create more instances to implement multi-level discriminator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: " + str(device))

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
        return None
    
    optimizer_D = torch.optim.Adam(model_D.parameters(), lr= args.learning_rate_D,betas=(0.9,0.99))


    if save_model_postfix != "":
        args.save_model_path = os.path.join(args.save_model_path, save_model_postfix)

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
        )

    # final test
    if args.mode != "train" or args.use_best:
        checkpoint_filename = os.path.join(args.save_model_path, "best.tar")
        logger.info(f"Performing final evaluation with the best model saved in the following checkpoint: {checkpoint_filename}")
        # Load Best before final evaluation
        if os.path.exists(checkpoint_filename):
            checkpoint = torch.load(checkpoint_filename)
            start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            logger.info(f"Loaded latest checkpoint that was at epoch n. {start_epoch}")

    return val(args, model, dataloader_val)


if __name__ == "__main__":
    
    logger.info("tg:Starting MEGA ADA")
    try:
        logger.info("tg:Starting task 3: ADA, GTA5 -> Cityscapes")
        precision_3, miou_3 = main("GTA5", "Cityscapes", "Cityscapes", save_model_postfix="3")
        logger.info(f"tg:3 Results: Precision={precision_3} Mean IoU={miou_3}")
    except Exception as e:
        logger.critical("tg:Error on 3", exc_info=e)

    
    try:
        logger.info("tg:Starting task 3: ADA, GTA5+aug -> Cityscapes")
        precision_3_aug, miou_3_aug = main("GTA5", "Cityscapes", "Cityscapes",augmentation=True, save_model_postfix="3_aug")
        logger.info(f"tg:3_aug Results: Precision={precision_3_aug} Mean IoU={miou_3_aug}")
    except Exception as e:
        logger.critical("tg:Error on 3_aug", exc_info=e)


# Doubts:

# use F.softmax or FN.softmax to compute output of models ?

# Training Generator loss: do we add the discriminator loss (peppe-sc) or not (Alessio) ?

# check from 272 to 318 if anything needs to be changed

# check from 548 to 556 if anything needs to be changed

# should we implement the multilevel discriminator (D1, D2, D3) ? -> multiple instances of discriminator, 
# each uses outputs from a different level of the NN ?

# we could hypertune lambda, as for the moment we only use lambda_d1

