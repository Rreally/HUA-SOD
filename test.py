import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import cv2
import math
import numpy as np
from dataset import build_loader
from configs.optimizer import build_optimizer
from utils import equi2pers, load_finetune, load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor
from utils import equi2panels
import torch.nn.functional as F
from networks.hivit_sod_revise import HiViTSOD
from configs.configloss import get_config
# from vis_features import show_feature
def mkdirs(path):
    try:
        os.makedirs(path)
    except:
        pass


class Saver(object):

    def __init__(self, save_dir, data_name, dataset):
        self.idx = -1

        self.dataname = data_name.name
        self.dataset = dataset

        self.save_dir = os.path.join(save_dir)
        if not os.path.exists(self.save_dir):
            mkdirs(self.save_dir)

    def save_samples(self, rgbs, gts, pred_salients, rgb_name):
        """
        Saves samples
        """
        rgbs = rgbs.cpu().numpy().transpose(0, 2, 3, 1)
        salient_preds = pred_salients.data.cpu().numpy()

        for i in range(rgbs.shape[0]):
            self.idx = self.idx + 1


            salient_pred = salient_preds[i][0]
            salient_pred = (salient_pred - salient_pred.min()) / (salient_pred.max() - salient_pred.min() + 1e-8)
            salient_pred = salient_pred.astype(np.float32)

            path = os.path.join(self.save_dir,  '%s' % (self.dataname[self.idx]) + '.png')
            cv2.imwrite(path, salient_pred * 255)

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)

    # easy config modification
    parser.add_argument('--batch_size', type=int, default=64, help="batch size for single GPU")
    parser.add_argument('--data_path', type=str, default='../data/', help='path to dataset')
    parser.add_argument("--dataset", default="360-SOD", type=str, help="dataset to train on.")
    parser.add_argument('--height', type=int, default=512, help='image height')
    parser.add_argument('--weight', type=int, default=1024, help='image height')
    parser.add_argument('--cache_mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--output', default='./pred_maps/preds_', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--model_path', default='/model/path', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--epoch', type=int, default=0, help='number of epoch')
    # distributed training
    parser.add_argument("--local_rank", type=str, default=0, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

def mkdirs(path):
    try:
        os.makedirs(path)
    except:
        pass

def main(args, config):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.local_rank
    model = HiViTSOD(img_size = (512, 1024))
    model.cuda()
    
    model_path = args.model_path + 'ckpt_epoch_' + str(args.epoch) + '.pth'

    model.load_state_dict(torch.load(model_path,map_location='cuda')['model'])

    _, test_dataset, _, test_loader= build_loader(args, config, isTest=True)

    model.eval()
    
    output = args.output + str(args.epoch) + '/'
    if not os.path.exists(output):
        mkdirs(output)
    save_dir = os.path.join(output, args.dataset+'-te')
    saver = Saver(save_dir, test_dataset, args.dataset)
    
    with torch.no_grad():
        for idx, (samples,  samples_hivit, targets, img_size, rgb_name) in enumerate(test_loader):
            # panels, _ = equi2panels(samples, I_rad=math.pi * 0.5, S_rad=math.pi * 0.25, N=8)
            # panels, _, _, _ = equi2pers(samples, nrows=4, fov=(80, 80), patch_size=(224, 224))
            # panels, _, _, _ = equi2pers(samples, nrows=4, fov=(80, 80), patch_size=(224, 224)) # B, C, H, W, N
            # _, xyz, uv, center_points = equi2pers(samples, nrows=4, fov=(80, 80), patch_size=(224//4, 224//4))
            panels, _, _, _ = equi2pers(samples, nrows=3, fov=(120, 120), patch_size=(256, 256)) # B, C, H, W, N
            _, xyz, uv, center_points = equi2pers(samples, nrows=3, fov=(120, 120), patch_size=(256//4, 256//4))
            if torch.cuda.is_available():
                samples = samples.cuda()
                samples_hivit = samples_hivit.cuda()
                panels = panels.cuda(non_blocking=True)
                
            outputs= model(samples, samples_hivit, panels, xyz, uv, center_points)
            
            #pred = torch.sigmoid(outputs[-1])
            #pred = torch.sigmoid(outputserpfusion)
            # pred_salient = F.interpolate(outputserpfusion, size=img_size, mode='bilinear', align_corners=True)
            pred_salient = F.interpolate(outputs[-1], size=img_size, mode='bilinear', align_corners=True)
            
            pred_salient = pred_salient.detach().cpu()
            saver.save_samples(samples, targets, pred_salient, rgb_name)

                

if __name__ == '__main__':
    args, config = parse_option()

    main(args, config)