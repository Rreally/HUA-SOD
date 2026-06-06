
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import cv2
import numpy as np
import math
import torch
import torch.nn.functional as F
import os
import torch.distributed as dist
from collections import OrderedDict
import numpy as np
from scipy.ndimage import map_coordinates

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

import math
from torch.utils import data as data
import numpy as np
import cv2
import torch


# generate patches in a closed-form
# the transformation and equation is referred from http://blog.nitishmutha.com/equirectangular/360degree/2017/06/12/How-to-project-Equirectangular-image-to-rectilinear-view.html
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def uv2xyz(uv):
    xyz = np.zeros((*uv.shape[:-1], 3), dtype = np.float32)
    xyz[..., 0] = np.multiply(np.cos(uv[..., 1]), np.sin(uv[..., 0]))
    xyz[..., 1] = np.multiply(np.cos(uv[..., 1]), np.cos(uv[..., 0]))
    xyz[..., 2] = np.sin(uv[..., 1])
    return xyz

def equi2pers(erp_img, fov, nrows, patch_size):
    bs, _, erp_h, erp_w = erp_img.shape
    height, width = pair(patch_size)
    fov_h, fov_w = pair(fov)
    FOV = torch.tensor([fov_w/360.0, fov_h/180.0], dtype=torch.float32)

    PI = math.pi
    PI_2 = math.pi * 0.5
    PI2 = math.pi * 2
    yy, xx = torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, width))
    screen_points = torch.stack([xx.flatten(), yy.flatten()], -1)
    
    if nrows==4:    
        num_rows = 4
        num_cols = [3, 6, 6, 3]
        phi_centers = [-67.5, -22.5, 22.5, 67.5]
    if nrows==6:    
        num_rows = 6
        num_cols = [3, 8, 12, 12, 8, 3]
        phi_centers = [-75.2, -45.93, -15.72, 15.72, 45.93, 75.2]
    if nrows==3:
        num_rows = 3
        num_cols = [3, 4, 3]
        phi_centers = [-60, 0, 60]     
    if nrows==5:
        num_rows = 5
        num_cols = [3, 6, 8, 6, 3]
        phi_centers = [-72.2, -36.1, 0, 36.1, 72.2]
            
    phi_interval = 180 // num_rows
    all_combos = []
    erp_mask = []
    for i, n_cols in enumerate(num_cols):
        for j in np.arange(n_cols):
            theta_interval = 360 / n_cols
            theta_center = j * theta_interval + theta_interval / 2

            center = [theta_center, phi_centers[i]]
            all_combos.append(center)
            up = phi_centers[i] + phi_interval / 2
            down = phi_centers[i] - phi_interval / 2
            left = theta_center - theta_interval / 2
            right = theta_center + theta_interval / 2
            up = int((up + 90) / 180 * erp_h)
            down = int((down + 90) / 180 * erp_h)
            left = int(left / 360 * erp_w)
            right = int(right / 360 * erp_w)
            mask = np.zeros((erp_h, erp_w), dtype=int)
            mask[down:up, left:right] = 1
            erp_mask.append(mask)
    all_combos = np.vstack(all_combos) 
    shifts = np.arange(all_combos.shape[0]) * width
    shifts = torch.from_numpy(shifts).float()
    erp_mask = np.stack(erp_mask)
    erp_mask = torch.from_numpy(erp_mask).float()
    num_patch = all_combos.shape[0]

    center_point = torch.from_numpy(all_combos).float()  # -180 to 180, -90 to 90
    center_point[:, 0] = (center_point[:, 0]) / 360  #0 to 1
    center_point[:, 1] = (center_point[:, 1] + 90) / 180  #0 to 1

    cp = center_point * 2 - 1
    center_p = cp.clone()
    cp[:, 0] = cp[:, 0] * PI
    cp[:, 1] = cp[:, 1] * PI_2
    cp = cp.unsqueeze(1)
    convertedCoord = screen_points * 2 - 1
    convertedCoord[:, 0] = convertedCoord[:, 0] * PI
    convertedCoord[:, 1] = convertedCoord[:, 1] * PI_2
    convertedCoord = convertedCoord * (torch.ones(screen_points.shape, dtype=torch.float32) * FOV)
    convertedCoord = convertedCoord.unsqueeze(0).repeat(cp.shape[0], 1, 1)

    x = convertedCoord[:, :, 0]
    y = convertedCoord[:, :, 1]

    rou = torch.sqrt(x ** 2 + y ** 2)
    c = torch.atan(rou)
    sin_c = torch.sin(c)
    cos_c = torch.cos(c)
    lat = torch.asin(cos_c * torch.sin(cp[:, :, 1]) + (y * sin_c * torch.cos(cp[:, :, 1])) / rou)
    lon = cp[:, :, 0] + torch.atan2(x * sin_c, rou * torch.cos(cp[:, :, 1]) * cos_c - y * torch.sin(cp[:, :, 1]) * sin_c)
    lat_new = lat / PI_2 
    lon_new = lon / PI 
    lon_new[lon_new > 1] -= 2
    lon_new[lon_new<-1] += 2 

    lon_new = lon_new.view(1, num_patch, height, width).permute(0, 2, 1, 3).contiguous().view(height, num_patch*width)
    lat_new = lat_new.view(1, num_patch, height, width).permute(0, 2, 1, 3).contiguous().view(height, num_patch*width)
    grid = torch.stack([lon_new, lat_new], -1)
    grid = grid.unsqueeze(0).repeat(bs, 1, 1, 1).to(erp_img.device)

    pers = F.grid_sample(erp_img, grid, mode='bilinear', padding_mode='border', align_corners=True)
    pers = F.unfold(pers, kernel_size=(height, width), stride=(height, width))
    pers = pers.reshape(bs, -1, height, width, num_patch)
    pers = torch.stack([pers[:, :, :, :, i] for i in range(num_patch)], dim=0)
    pers = pers.view(bs * num_patch, -1, height, width)
    
    grid_tmp = torch.stack([lon, lat], -1)
    xyz = uv2xyz(grid_tmp)
    xyz = xyz.reshape(num_patch, height, width, 3).transpose(0, 3, 1, 2)
    xyz = torch.from_numpy(xyz).to(pers.device).contiguous()
    
    uv = grid[0, ...].reshape(height, width, num_patch, 2).permute(2, 3, 0, 1)
    uv = uv.contiguous()
    return pers, xyz, uv, center_p



def equi2panels(erp, I_rad, S_rad, N):
    bs, _, He, We = erp.shape
    I = int(I_rad / (2 * math.pi) * We)
    S = int(S_rad / (2 * math.pi) * We)

    offset = torch.arange(0, We, S) 
    j, i = torch.meshgrid(torch.arange(He), torch.arange(I), indexing='ij')
    v = ((j+0.5) / He - 0.5) * math.pi
    u = (i[None,...] + offset[:, None, None] + 0.5) / We * 2 * math.pi

    # 计算三维绝对坐标和相对坐标（几何特征）
    z = -torch.sin(v)
    c = torch.cos(v)
    y = c * torch.sin(u)
    x = c * torch.cos(u)
    u_r = (i+0.5) / We * 2 * math.pi
    y_r = c * torch.sin(u_r)
    x_r = c * torch.cos(u_r)

    # 把绝对位置x, y, z和相对位置x_r, y_r五个属性堆叠起来
    a1 = torch.stack([z[None, ...], x_r[None, ...], y_r[None, ...]], axis=1).repeat(N, 1, 1, 1)
    geo_feat = torch.cat([x[:, None, ...], y[:, None, ...], a1], axis=1)
    # print(geo_feat.shape)  # (N, 5, He, I), 与batch size大小无关 

    # 计算panel中每个像素对应原ERP图像中的采样点位置
    grid_h = v * 2.0 / math.pi
    grid_w = u / math.pi - 1
    grid_w[grid_w > 1] -= 2
    grid_w = grid_w.permute(1,0,2).contiguous().view(He,N * I)
    grid = torch.stack((grid_w, grid_h.repeat(1, N)), dim=-1)
    grid = grid.unsqueeze(0).repeat(bs, 1, 1, 1).to(erp.device)

    # 重采样，以从原ERP图像中得到N个panel图像
    panels = F.grid_sample(erp, grid=grid, mode='bilinear', padding_mode='border', align_corners=True)
    panels = F.unfold(panels, kernel_size=(He, I), stride=(He, I))
    panels = panels.reshape(bs, -1, He, I, N)
    panels = torch.stack([panels[:, :, :, :, i] for i in range(N)], dim=0)
    # print(bs, N)
    panels = panels.view(bs * N, -1, He, I)

    return panels, geo_feat


def total_loss(pred, mask):
    return F.binary_cross_entropy_with_logits(pred, mask, reduction='sum') + structure_loss(pred, mask)


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def load_finetune(config, model, logger):
    logger.info(f"==============> Finetuning form {config.MODEL.FINETUNE}....................")
    checkpoint = torch.load(config.MODEL.FINETUNE, map_location='cpu')
    state_dict = OrderedDict()
    model_state_dict = model.state_dict()
    for k, v in checkpoint['model'].items():
        if k in ['relative_position_index']:
            continue
        elif k.endswith('patch_embed.proj.weight') and k in model_state_dict:
            S1 = v.size(-1)
            S2 = model_state_dict[k].size(-1)
            if S1 != S2:
                v = F.interpolate(
                    v,
                    scale_factor=(S2 / S1, S2 / S1),
                    mode='bicubic',
                )
        elif k.endswith('_pos_embed') and k in model_state_dict:
            S1 = int(math.sqrt(v.size(1)))
            S2 = int(math.sqrt(model_state_dict[k].size(1)))
            if S1 != S2:
                v = F.interpolate(
                    v.reshape(1, S1, S1, -1).permute(0, 3, 1, 2),
                    scale_factor=(S2 / S1, S2 / S1),
                    mode='bicubic',
                ).flatten(2).transpose(1, 2)
        elif k.endswith('.relative_position_bias_table') and k in model_state_dict:
            S1 = int(math.sqrt(v.size(0)))
            S2 = int(math.sqrt(model_state_dict[k].size(0)))
            if S1 != S2:
                v = F.interpolate(
                    v.reshape(1, S1, S1, -1).permute(0, 3, 1, 2),
                    scale_factor=(S2 / S1, S2 / S1),
                    mode='bicubic',
                ).flatten(2).transpose(1, 2)[0]
        state_dict[k] = v
    msg = model.load_state_dict(state_dict, strict=False)
    logger.info(msg)


def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'amp' in checkpoint and config.AMP_OPT_LEVEL != "O0" and checkpoint['config'].AMP_OPT_LEVEL != "O0":
            amp.load_state_dict(checkpoint['amp'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    if config.AMP_OPT_LEVEL != "O0":
        save_state['amp'] = amp.state_dict()

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

class EquirecRotatedAug:
    def __init__(self, equ_h, equ_w, theta, ax):
        """
        self.equ_h, self.equ_w: Image size
        self.theta            : rotation angle
        self.ax               : rotate around which axis
        """
        self.equ_h = equ_h
        self.equ_w = equ_w
        self.theta = theta
        self.ax = ax

        self._xyz()
        # self.xyzRotate()
        # self.xyz2coor()

    def _xyz(self):
        """ self.xyz: get xyz coordinates of ERP """
        self.xyz = np.ones((self.equ_h, self.equ_w, 3), np.float32)
        u = np.linspace(-np.pi, np.pi, num=self.equ_w, dtype=np.float32)
        v = np.linspace(np.pi, -np.pi, num=self.equ_h, dtype=np.float32) / 2
        self.grid = np.stack(np.meshgrid(u, v), axis=-1)
        self.xyz[:, :, 0] = np.cos(self.grid[:, :, 1]) * np.sin(self.grid[:, :, 0])
        self.xyz[:, :, 1] = np.sin(self.grid[:, :, 1])
        self.xyz[:, :, 2] = np.cos(self.grid[:, :, 1]) * np.cos(self.grid[:, :, 0])

    def xyzRotate(self):
        """ rotate xyz """
        theta = -self.theta * np.pi / 180
        R = rotation_matrix(theta, self.ax)
        self.xyz = self.xyz.dot(R)

    def xyz2coor(self):
        """ xyz coordinates to uv coordinates """
        x, y, z = np.split(self.xyz, 3, axis=-1)
        u = np.arctan2(x, z)
        c = np.sqrt(x ** 2 + z ** 2)
        v = np.arctan2(y, c)
        self.coor_x = (u / (2 * np.pi) + 0.5) * self.equ_w - 0.5
        self.coor_y = (-v / np.pi + 0.5) * self.equ_h - 0.5

    def sample_equiRec(self, e_img, order=0):
        pad_u = np.roll(e_img[[0]], self.equ_w // 2, 1)
        pad_d = np.roll(e_img[[-1]], self.equ_w // 2, 1)
        e_img = np.concatenate([e_img, pad_d, pad_u], 0)
        return map_coordinates(e_img, [self.coor_y, self.coor_x], order=order, mode='wrap')[..., 0]

    def run(self, equ_img, equ_gt, theta):
        self.theta = theta
        self.xyzRotate()
        self.xyz2coor()
        h, w = equ_img.shape[:2]
        equ_gt = equ_gt[..., np.newaxis]
        if h != self.equ_h or w != self.equ_w:
            equ_img = cv2.resize(equ_img, (self.equ_w, self.equ_h))
            if equ_gt is not None:
                equ_gt = cv2.resize(equ_gt, (self.equ_w, self.equ_h), interpolation=cv2.INTER_NEAREST)
        # print("equ_img",equ_img.shape)
        equ_img_rotated = np.stack([self.sample_equiRec(equ_img[..., i], order=1) for i in range(equ_img.shape[2])],
                                   axis=-1)

        if equ_gt is not None:
            # print(equ_gt.shape)
            equ_gt_rotated = np.stack([self.sample_equiRec(equ_gt[..., i], order=1) for i in range(equ_gt.shape[2])],
                                      axis=-1)
            # print(np.squeeze(equ_gt_rotated, axis=-1).shape)
        if equ_gt is not None:
            return equ_img_rotated, np.squeeze(equ_gt_rotated, axis=-1)
        else:
            return equ_img_rotated


def rotation_matrix(theta, ax):
    
    ax = np.array(ax)
    assert len(ax.shape) == 1 and ax.shape[0] == 3
    ax = ax / np.sqrt((ax ** 2).sum())
    R = np.diag([np.cos(theta)] * 3)
    R = R + np.outer(ax, ax) * (1.0 - np.cos(theta))
    ax = ax * np.sin(theta)
    R = R + np.array([[0, -ax[2], ax[1]],
                      [ax[2], 0, -ax[0]],
                      [-ax[1], ax[0], 0]])
    return R


import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import cv2
import time
from os import makedirs
from os.path import join, exists
#from equi2pers_v3 import equi2pers
import time
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def pers2equi(pers_img, fov, nrows, patch_size, erp_size, layer_name):
    bs = pers_img.shape[0]
    channel = pers_img.shape[1]
    device=pers_img.device
    height, width = pair(patch_size)
    fov_h, fov_w = pair(fov)
    erp_h, erp_w = pair(erp_size)
    n_patch = pers_img.shape[-1]     
    grid_dir = './grid'
    if not exists(grid_dir):
        makedirs(grid_dir)
    grid_file = join(grid_dir, layer_name + '.pth')  
    
    if not exists(grid_file):  
        FOV = torch.tensor([fov_w/360.0, fov_h/180.0], dtype=torch.float32)

        PI = math.pi
        PI_2 = math.pi * 0.5
        PI2 = math.pi * 2

        if nrows==4:    
            num_rows = 4
            num_cols = [3, 6, 6, 3]
            phi_centers = [-67.5, -22.5, 22.5, 67.5]
        if nrows==6:    
            num_rows = 6
            num_cols = [3, 8, 12, 12, 8, 3]
            phi_centers = [-75.2, -45.93, -15.72, 15.72, 45.93, 75.2]
        if nrows==3:
            num_rows = 3
            num_cols = [3, 4, 3]
            phi_centers = [-59.6, 0, 59.6]
        if nrows==5:
            num_rows = 5
            num_cols = [3, 6, 8, 6, 3]
            phi_centers = [-72.2, -36.1, 0, 36.1, 72.2] 
        phi_interval = 180 // num_rows
        all_combos = []

        for i, n_cols in enumerate(num_cols):
            for j in np.arange(n_cols):
                theta_interval = 360 / n_cols
                theta_center = j * theta_interval + theta_interval / 2

                center = [theta_center, phi_centers[i]]
                all_combos.append(center)
                
                
        all_combos = np.vstack(all_combos) 
        n_patch = all_combos.shape[0]
        
        center_point = torch.from_numpy(all_combos).float()  # -180 to 180, -90 to 90
        center_point[:, 0] = (center_point[:, 0]) / 360  #0 to 1
        center_point[:, 1] = (center_point[:, 1] + 90) / 180  #0 to 1

        cp = center_point * 2 - 1
        cp[:, 0] = cp[:, 0] * PI
        cp[:, 1] = cp[:, 1] * PI_2
        cp = cp.unsqueeze(1)
        
        lat_grid, lon_grid = torch.meshgrid(torch.linspace(-PI_2, PI_2, erp_h), torch.linspace(-PI, PI, erp_w))
        lon_grid = lon_grid.float().reshape(1, -1)#.repeat(num_rows*num_cols, 1)
        lat_grid = lat_grid.float().reshape(1, -1)#.repeat(num_rows*num_cols, 1) 
        cos_c = torch.sin(cp[..., 1]) * torch.sin(lat_grid) + torch.cos(cp[..., 1]) * torch.cos(lat_grid) * torch.cos(lon_grid - cp[..., 0])
        new_x = (torch.cos(lat_grid) * torch.sin(lon_grid - cp[..., 0])) / cos_c
        new_y = (torch.cos(cp[..., 1])*torch.sin(lat_grid) - torch.sin(cp[...,1])*torch.cos(lat_grid)*torch.cos(lon_grid-cp[...,0])) / cos_c
        new_x = new_x / FOV[0] / PI   # -1 to 1
        new_y = new_y / FOV[1] / PI_2
        cos_c_mask = cos_c.reshape(n_patch, erp_h, erp_w)
        cos_c_mask = torch.where(cos_c_mask > 0, 1, 0)
        
        w_list = torch.zeros((n_patch, erp_h, erp_w, 4), dtype=torch.float32)

        new_x_patch = (new_x + 1) * 0.5 * height
        new_y_patch = (new_y + 1) * 0.5 * width 
        new_x_patch = new_x_patch.reshape(n_patch, erp_h, erp_w)
        new_y_patch = new_y_patch.reshape(n_patch, erp_h, erp_w)
        mask = torch.where((new_x_patch < width) & (new_x_patch > 0) & (new_y_patch < height) & (new_y_patch > 0), 1, 0)
        mask *= cos_c_mask

        x0 = torch.floor(new_x_patch).type(torch.int64)
        x1 = x0 + 1
        y0 = torch.floor(new_y_patch).type(torch.int64)
        y1 = y0 + 1

        x0 = torch.clamp(x0, 0, width-1)
        x1 = torch.clamp(x1, 0, width-1)
        y0 = torch.clamp(y0, 0, height-1)
        y1 = torch.clamp(y1, 0, height-1)

        wa = (x1.type(torch.float32)-new_x_patch) * (y1.type(torch.float32)-new_y_patch)
        wb = (x1.type(torch.float32)-new_x_patch) * (new_y_patch-y0.type(torch.float32))
        wc = (new_x_patch-x0.type(torch.float32)) * (y1.type(torch.float32)-new_y_patch)
        wd = (new_x_patch-x0.type(torch.float32)) * (new_y_patch-y0.type(torch.float32))

        wa = wa * mask.expand_as(wa)
        wb = wb * mask.expand_as(wb)
        wc = wc * mask.expand_as(wc)
        wd = wd * mask.expand_as(wd)

        w_list[..., 0] = wa
        w_list[..., 1] = wb
        w_list[..., 2] = wc
        w_list[..., 3] = wd

        save_file = {'x0':x0, 'y0':y0, 'x1':x1, 'y1':y1, 'w_list': w_list, 'mask':mask}
        torch.save(save_file, grid_file)
    else:
        # the online merge really takes time
        # pre-calculate the grid for once and use it during training
        load_file = torch.load(grid_file)
        #print('load_file')
        x0 = load_file['x0']
        y0 = load_file['y0']
        x1 = load_file['x1']
        y1 = load_file['y1']
        w_list = load_file['w_list']
        mask = load_file['mask']

    w_list = w_list.to(device)
    mask = mask.to(device)
    z = torch.arange(n_patch)
    z = z.reshape(n_patch, 1, 1)
    #start = time.time()
    Ia = pers_img[:, :, y0, x0, z]
    Ib = pers_img[:, :, y1, x0, z]
    Ic = pers_img[:, :, y0, x1, z]
    Id = pers_img[:, :, y1, x1, z]
    #print(time.time() - start)
    output_a = Ia * mask.expand_as(Ia)
    output_b = Ib * mask.expand_as(Ib)
    output_c = Ic * mask.expand_as(Ic)
    output_d = Id * mask.expand_as(Id)

    output_a = output_a.permute(0, 1, 3, 4, 2)
    output_b = output_b.permute(0, 1, 3, 4, 2)
    output_c = output_c.permute(0, 1, 3, 4, 2)
    output_d = output_d.permute(0, 1, 3, 4, 2)   
    #print(time.time() - start)
    w_list = w_list.permute(1, 2, 0, 3)
    w_list = w_list.flatten(2)
    w_list *= torch.gt(w_list, 1e-5).type(torch.float32)
    w_list = F.normalize(w_list, p=1, dim=-1).reshape(erp_h, erp_w, n_patch, 4)
    w_list = w_list.unsqueeze(0).unsqueeze(0)
    output = output_a * w_list[..., 0] + output_b * w_list[..., 1] + \
        output_c * w_list[..., 2] + output_d * w_list[..., 3]
    img_erp = output.sum(-1) 

    return img_erp

if __name__ == "__main__":
    img = cv2.imread('/home/ResHiViT_exp/data/360-SOD/360-SOD-te/imgs/wild-360_6Scd9baRjf8_2_4.jpg', cv2.IMREAD_COLOR)
    img_size = (512, 256)
    img = cv2.resize(img, img_size, interpolation=cv2.INTER_CUBIC)
    print("img shape", img.shape)
    img_new = img.astype(np.float32) 
    img_new = np.transpose(img_new, [2, 0, 1])
    img_new = torch.from_numpy(img_new)
    img_new = img_new.unsqueeze(0)
    fov = (120, 120)
    
    pers, _, _, _ = equi2pers(img_new, nrows=3, fov=fov, patch_size=(256, 256))
    
    print("pers shape",pers.shape)
    pers = torch.stack(torch.split(pers, 1, dim=0), dim=-1)
    # print(pers.shape)
    # pers = F.unfold(pers, kernel_size=64, stride=64)
    # pers = pers.reshape(1, 3, 256, 256, -1)

    erp = pers2equi(pers, nrows=4, fov=fov, patch_size=(112, 112), erp_size=(256, 512), layer_name='pred')
    print("pers2erp shape",erp.shape)
    n_patch = pers.shape[-1]
    img_erp_int = erp[0, ...].permute(1, 2, 0).numpy()
    img_erp_int = img_erp_int #* 255
    img_erp_int = img_erp_int.astype(np.uint8)
    cv2.imwrite('interp_erp.png', img_erp_int)

# if __name__ == '__main__':
#     img = cv2.imread('/home/ResHiViT_exp/data/360-SOD/360-SOD-te/imgs/360-saliency-dataset_220_14.jpg', cv2.IMREAD_COLOR)
#     img_new = img.astype(np.float32) 
#     img_new = np.transpose(img_new, [2, 0, 1])
#     img_new = torch.from_numpy(img_new)
#     img_new = img_new.unsqueeze(0)
#     pers = equi2pers(img_new, nrows=3, fov=(80, 80), patch_size=(224, 224))
#     pers = pers[0].numpy()
#     print(pers.shape[0])
#     for i in range(pers.shape[0]):
#         print(i)
#         per = pers[i].transpose(1, 2, 0).astype(np.uint8)
#         cv2.imwrite('pers'+str(i)+'.png', per)
        
