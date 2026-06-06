# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# python train.py --cfg configs/hivit_base_224.yaml --data-pathta-path ../../data/  --batch-size 64
import os
import time

import argparse
import datetime
import numpy as np
import math
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from timm.utils import AverageMeter

from configs.lr_scheduler import build_scheduler
from configs.optimizer import build_optimizer
from configs.logger import create_logger
from configs.options import parse_option

# from networks.hivit_sod_net import HiViTSOD
from networks.hivit_sod_revise import HiViTSOD
from dataset import build_loader
from utils import equi2pers, load_checkpoint, save_checkpoint, auto_resume_helper, total_loss

from tensorboardX import SummaryWriter

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def main(args, config):
    dataset_train, _, data_loader_train, _ = build_loader(args, config)
    model = HiViTSOD(img_size=(512, 1024))
    model.load_pre('/data/pretrainedModels/mae_hivit_base_1600ep_ft100ep.pth')

    print_network(model, "Ours")
    if torch.cuda.is_available():
        model.cuda()

    optimizer = build_optimizer(config, model)
    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)

    model_without_ddp = model

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    criterion = torch.nn.BCEWithLogitsLoss()

    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
    
    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        # data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, lr_scheduler)
        if epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, lr_scheduler):
    model.train()
    optimizer.zero_grad()
    writer1 = SummaryWriter('runs/nowarmup')
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    mae_meter = AverageMeter()
    start = time.time()
    end = time.time()
    for idx, (samples, samples_hivit, targets, _, rgb_name) in enumerate(data_loader):
        # todo: 将ERP图转换为Panel

        panels, _, _, _ = equi2pers(samples, nrows=3, fov=(120, 120), patch_size=(256, 256))  # B, C, H, W, N
        _, xyz, uv, center_points = equi2pers(samples, nrows=3, fov=(120, 120), patch_size=(256 // 4, 256 // 4))

        if torch.cuda.is_available():
            samples = samples.cuda(non_blocking=True)
            samples_hivit = samples_hivit.cuda(non_blocking=True)
            panels = panels.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        outputs = model(samples, samples_hivit, panels, xyz, uv, center_points)
        #loss5 = total_loss(outputs[5], targets.float())
        #loss4 = total_loss(outputs[4], targets.float())
        #loss3 = total_loss(outputs[3], targets.float())
        #loss2 = total_loss(outputs[2], targets.float())
        #loss1 = total_loss(outputs[1], targets.float())
        #loss0 = total_loss(outputs[0], targets.float())
        loss5 = F.binary_cross_entropy_with_logits(outputs[5], targets, reduction='sum') + structure_loss(outputs[5], targets)
        loss4 = F.binary_cross_entropy_with_logits(outputs[4], targets, reduction='sum') + structure_loss(outputs[4], targets)
        loss3 = F.binary_cross_entropy_with_logits(outputs[3], targets, reduction='sum') + structure_loss(outputs[3], targets)
        loss2 = F.binary_cross_entropy_with_logits(outputs[2], targets, reduction='sum') + structure_loss(outputs[2], targets)
        loss1 = F.binary_cross_entropy_with_logits(outputs[1], targets, reduction='sum') + structure_loss(outputs[1], targets)
        loss0 = F.binary_cross_entropy_with_logits(outputs[0], targets, reduction='sum') + structure_loss(outputs[0], targets)
        loss = (loss5 + loss4 + loss3 + loss2 + loss1 + loss0) / 6.

        mae = F.l1_loss(outputs[-1],targets)
        # L1 正则化
        #re_loss = 0
        #for param in model.parameters():
        #    re_loss += torch.sum(torch.abs(param))
        #loss = loss + 0.01 * re_loss

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)

        optimizer.step()
        lr_scheduler.step_update(epoch * num_steps + idx)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        mae_meter.update(mae.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        writer1.add_scalar('Loss/train', loss_meter.val, epoch * num_steps + idx)
        writer1.add_scalar('MAE/train', mae_meter.val, epoch * num_steps + idx)

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'MAE {mae_meter.val:.4f} ({mae_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

        # 在每个 epoch 结束时记录平均的 loss 和 MAE
    writer1.add_scalar('Loss/epoch_avg', loss_meter.avg, epoch)
    writer1.add_scalar('MAE/epoch_avg', mae_meter.avg, epoch)
    writer1.close()


def print_network(model, name):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(name)
    print("The number of parameters: {}".format(num_params))

def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).sum()

if __name__ == '__main__':
    args, config = parse_option()

    world_size = -1
    if torch.cuda.is_available():
        torch.cuda.set_device(config.LOCAL_RANK)

    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")

    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")

    main(args, config)
