from __future__ import print_function
import os
from utils import equi2pers
import cv2
import numpy as np
import random
import math
import torch
import tqdm
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms
from torch.utils.data import DataLoader
import sys
# from utils import equi2panels as e2p

def build_loader(args, config, isTest=False):

    # args.data_path '/data/PISOD/data/'  args.dataset '360-SOD'
    fpath = os.path.join(args.data_path, args.dataset, "{}-{}.txt")
    if isTest is False:
        train_file_list = fpath.format(args.dataset, "tr")
        dataset_train = Dataset(args.data_path, args.dataset, train_file_list, args.height, args.weight)
        data_loader_train = DataLoader(dataset_train, args.batch_size, True, num_workers=config.DATA.NUM_WORKERS , pin_memory=False, drop_last=True)

    test_file_list = fpath.format(args.dataset, "te")
    dataset_val = Dataset(args.data_path, args.dataset, test_file_list, args.height, args.weight)
    data_loader_val = DataLoader(dataset_val, args.batch_size, False, num_workers=config.DATA.NUM_WORKERS ,
                             pin_memory=False, drop_last=False)
    if isTest is True:
        return None, dataset_val, None, data_loader_val

    return dataset_train, dataset_val, data_loader_train, data_loader_val

def read_list(list_file):
    rgb_gt_list = []
    rgb_list = []
    with open(list_file) as f:
        lines = f.readlines()
        for line in lines:
            rgb_gt_list.append(line.strip().split(" "))
            rgb_list.append(line.strip().split(" ")[0].split('/')[-1].split('.')[0])
    return rgb_gt_list, rgb_list


class Dataset(data.Dataset):
    def __init__(self, root_dir, dataset, list_file, height=512, width=1024):
        
        self.dataset = dataset
        self.root_dir = root_dir + self.dataset
        self.rgb_gt_list, self.name = read_list(list_file)
        self.w = width
        self.h = height

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.rgb_gt_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        rgb_name = os.path.join(self.root_dir, self.rgb_gt_list[idx][0])
        rgb = cv2.imread(rgb_name)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        img_size = (rgb.shape[0], rgb.shape[1])
        rgb = cv2.resize(rgb, dsize=(self.w, self.h), interpolation=cv2.INTER_CUBIC)
        rgb_hivit = cv2.resize(rgb, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        
        gt_name = os.path.join(self.root_dir, self.rgb_gt_list[idx][1])
        gt = cv2.imread(gt_name, -1)
        gt = cv2.resize(gt, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
        gt = gt.astype(np.float32) / 255
        
        rgb = self.to_tensor(rgb.copy())
        rgb = self.normalize(rgb)
        rgb_hivit = self.to_tensor(rgb_hivit.copy())
        rgb_hivit = self.normalize(rgb_hivit)
        gt = torch.from_numpy(np.expand_dims(gt, axis=0))

        return rgb, rgb_hivit, gt, img_size, rgb_name


if __name__ == '__main__':
    device = torch.device("cuda")
    
    fpath = os.path.join('/home/ResHiViT_exp/data/', '360-SOD', "{}-{}.txt")
    train_file_list = fpath.format('360-SOD', "tr")
    train_dataset = Dataset('/home/ResHiViT_exp/data/', '360-SOD', train_file_list, 512, 1024)
    train_loader = DataLoader(train_dataset, 2, True, num_workers=4, pin_memory=False, drop_last=True)
    # print(fpath, train_file_list)
    pbar = tqdm.tqdm(train_loader)
    for batch_idx, (rgb, rgb_hivit, gt, _, _) in enumerate(pbar):
        
        # panels_inputs, geo_feat = e2p(rgb, I_rad = (math.pi * 0.25), S_rad = (math.pi * 0.0625), N = 32)
        high_res_patch, _, _, _ = equi2pers(rgb, nrows=3, fov=(52, 52), patch_size=(224, 224))
    