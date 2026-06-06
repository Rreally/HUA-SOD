import math
import os
import cv2
from utils import pers2equi
from .resnet import resnet50
import torch
import torch.nn as nn
import numpy as np
from functools import partial
import torch.nn.functional as F
from .dcn import DCNv2, get_condition

from .layers import Att, Pred_Layer, BAM, MBAM, FFM, FeatureProcessBlock, ScConv
from .hivit import HiViT
import sys

sys.path.append('../')


# 不确定性网络
class Uncertainty(nn.Module):
    def __init__(self):
        super(Uncertainty, self).__init__()
        self.conv1 = nn.Sequential(nn.ReLU(),
                                   nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.ReLU(),
                                   nn.Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.ReLU(),
                                   nn.Conv2d(16, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.Sigmoid())
        self.conv2 = nn.Sequential(nn.ReLU(),
                                   nn.Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.ReLU(),
                                   nn.Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.ReLU(),
                                   nn.Conv2d(16, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.Sigmoid())
        self.conv3 = nn.Sequential(nn.ReLU(),
                                   nn.Conv2d(512, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.ReLU(),
                                   nn.Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.ReLU(),
                                   nn.Conv2d(16, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.Sigmoid())

    def forward(self, x):
        return [self.conv1(x[0]), self.conv2(x[1]), self.conv3(x[2])]


class HiViTSOD(nn.Module):
    def __init__(self, img_size=(512, 1024)):
        super(HiViTSOD, self).__init__()

        # ================ E2P分支 ================
        # E2P分支encoder-ResNet50
        self.encoder = resnet50(pretrained=True)
        self.uncertain_net = Uncertainty()

        # patch 基本属性
        self.I_rad = math.pi * 0.5
        self.S_rad = math.pi * 0.25
        self.N = 8

        # 位置信息处理-MLP
        self.mlp_points = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # 注意力机制
        self.att0 = Att(dim=64)
        self.att1 = Att(dim=256)
        self.att2 = Att(dim=512)
        self.att3 = Att(dim=1024)
        self.att4 = Att(dim=2048)

        # ================ ERP分支 ================
        # ERP分支encoder-HiViT
        self.equi_encoder = HiViT(img_size=224, patch_size=16, inner_patches=4, in_chans=3, num_classes=1000,
                                  embed_dim=512, depths=[2, 2, 20], num_heads=8, stem_mlp_ratio=3., mlp_ratio=4.,
                                  qkv_bias=True, qk_scale=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.5,
                                  norm_layer=partial(nn.LayerNorm, eps=1e-6), ape=True, rpe=True, patch_norm=True,
                                  use_checkpoint=True,
                                  kernel_size=None, pad_size=None)

        # ================ Decoder ================
        self.toplayer = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(2048, 32, kernel_size=5, stride=1, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # 频域傅里叶卷积
        self.fsfc0 = FeatureProcessBlock(128, 128)
        self.fsfc1 = FeatureProcessBlock(256, 256)
        self.fsfc2 = FeatureProcessBlock(512, 512)
        self.conv_ffc1 = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_ffc2 = nn.Conv2d(640, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_ffc3 = nn.Conv2d(1280, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.convtp1 = nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.convtp2 = nn.Conv2d(1280, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.convtp3 = nn.Conv2d(2560, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.converp_f = nn.Sequential(nn.Conv2d(512, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.ReLU(),
                                   nn.Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.ReLU(),
                                   nn.Conv2d(16, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.convtp_f = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.ReLU(),
                                   nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.ReLU(),
                                   nn.Conv2d(256, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))


        # E&P特征交互
        self.epfusion0 = FFM(512, 128)
        self.epfusion1 = FFM(1024, 256)
        self.epfusion2 = FFM(2048, 512)

        self.rgb_global = Pred_Layer(32)
        self.bams = nn.ModuleList([
            BAM(64),
            BAM(256),
            MBAM(128),
            MBAM(256),
            MBAM(512),
        ])

    def _upsample_add(self, x, y):
        [_, _, H, W] = y.size()
        return F.interpolate(
            x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, samples, samples_hivit, pers, xyz, uv, center_points):

        B, C, H_e, W_e = samples.shape
        BN, C, H_p, W_p = pers.shape
        device = samples.device

        # ============== E2P分支编码 ==============
        # 几何位置信息获取
        rho = torch.ones((uv.shape[0], 1, H_p // 4, W_p // 4), dtype=torch.float32, device=device)
        center_points = center_points.to(device)
        center_points = center_points.reshape(-1, 2, 1, 1).repeat(1, 1, H_p // 4, W_p // 4)
        n_patch = pers.shape[-1]
        new_xyz = torch.cat([center_points, rho, center_points], 1)
        point_feat = self.mlp_points(new_xyz.contiguous())
        point_feat = torch.cat([point_feat] * B, dim=0)

        # Encoder: ResNet50
        """res_stage0"""
        x = self.encoder.conv1(pers)
        x = self.encoder.relu(self.encoder.bn1(x))
        feat0 = x
        feat0 = self.att0(feat0, B)
        x = self.encoder.maxpool(x)
        """res_stage1"""
        res_feat1 = self.encoder.layer1(x)
        feat1 = self.att1(res_feat1, B)
        feat1 = feat1 + point_feat
        """res_stage2"""
        res_feat2 = self.encoder.layer2(res_feat1)
        feat2 = self.att2(res_feat2, B)
        """res_stage3"""
        res_feat3 = self.encoder.layer3(res_feat2)
        feat3 = self.att3(res_feat3, B)
        """res_stage4"""
        res_feat4 = self.encoder.layer4(res_feat3)
        feat4 = self.att4(res_feat4, B)

        feat5 = self.toplayer(feat4)
        feats = [feat0, feat1, feat2, feat3, feat4, feat5]

        p2e_feat = []
        for i in range(len(feats)):
            k = 2 ** (i + 1)
            patch_size = (H_p // k, W_p // k)
            erp_size = (H_e // k, W_e // k)
            layer_name = 'pred_' + str(erp_size[0]) + 'x' + str(erp_size[1])
            p2e_feat.append(torch.stack(torch.split(feats[i], samples.shape[0], dim=0), dim=-1))
            p2e_feat[i] = pers2equi(p2e_feat[i], nrows=3, fov=(120, 120), patch_size=patch_size, erp_size=erp_size,
                                    layer_name=layer_name)

        # ============== ERP分支编码 ==============
        # Encoder: HiVit，输入size为224x224
        equi_feat_list = self.equi_encoder(samples_hivit)
        # print(equi_feat_list[0].shape) # torch.Size([2, 128, 56, 56])
        # print(equi_feat_list[1].shape) # torch.Size([2, 256, 28, 28])
        # print(equi_feat_list[2].shape) # torch.Size([2, 512, 14, 14])
        equi_feat_list[0] = F.interpolate(equi_feat_list[0], size=(64, 128), mode='bilinear', align_corners=True)
        equi_feat_list[1] = F.interpolate(equi_feat_list[1], size=(32, 64), mode='bilinear', align_corners=True)
        equi_feat_list[2] = F.interpolate(equi_feat_list[2], size=(16, 32), mode='bilinear', align_corners=True)

        # Uncertainty
        sat_uncer_list = self.uncertain_net(equi_feat_list)
        sat_uncer_list0 = sat_uncer_list[0]
        sat_uncer_list1 = F.interpolate(sat_uncer_list[1], size=(64, 128), mode='bilinear', align_corners=True)
        sat_uncer_list2 = F.interpolate(sat_uncer_list[2], size=(64, 128), mode='bilinear', align_corners=True)
        #print('sat_uncer_list0:',sat_uncer_list0.size())
        #print('sat_uncer_list1:',sat_uncer_list1.size())
        #print('sat_uncer_list2:',sat_uncer_list2.size())
        # print(sat_uncer_list[0].shape) # torch.Size([2, 1, 64, 128])
        # print(sat_uncer_list[1].shape) # torch.Size([2, 1, 32, 64])
        # print(sat_uncer_list[2].shape) # torch.Size([2, 1, 16, 32])
        #exit()
        # 频域傅里叶卷积
        equi_feat_list_ffc1 = self.fsfc0(equi_feat_list[0])
        equi_feat_list_ffc2 = self.fsfc1(equi_feat_list[1])
        equi_feat_list_ffc3 = self.fsfc2(equi_feat_list[2])

        equi_feat_list1 = torch.cat([equi_feat_list[0], equi_feat_list_ffc1], 1)
        equi_feat_list2 = torch.cat([equi_feat_list[1], equi_feat_list_ffc2], 1)
        equi_feat_list3 = torch.cat([equi_feat_list[2], equi_feat_list_ffc3], 1)
        # print(equi_feat_list[0].shape) # torch.Size([2, 128, 64, 128])
        # print(equi_feat_list[1].shape) # torch.Size([2, 256, 32, 64])
        # print(equi_feat_list[2].shape) # torch.Size([2, 512, 16, 32])
        #print('equi_feat_list_ffc1:',equi_feat_list_ffc1.size())
        #print('equi_feat_list_ffc3:',equi_feat_list_ffc3.size())
        #print('equi_feat_list1:',equi_feat_list1.size())
        #print('equi_feat_list3:',equi_feat_list3.size())
        #exit()
        equi_feat_list11 = self.conv_ffc1(equi_feat_list1)
        #print('equi_feat_list11:',equi_feat_list11.size())
        equi_feat_list2 = F.interpolate(equi_feat_list2, size=(64, 128), mode='bilinear', align_corners=True)
        equi_feat_list22 = torch.cat([equi_feat_list11, equi_feat_list2], 1)
        equi_feat_list22 = self.conv_ffc2(equi_feat_list22)

        equi_feat_list3 = F.interpolate(equi_feat_list3, size=(64, 128), mode='bilinear', align_corners=True)
        equi_feat_list33 = torch.cat([equi_feat_list22, equi_feat_list3], 1)
        equi_feat_list33 = self.conv_ffc3(equi_feat_list33)

        pred_erp = self.converp_f(equi_feat_list33)
        pred_erp = torch.sigmoid(F.interpolate(pred_erp,
                          size=(H_e, W_e),
                          mode='bilinear',
                          align_corners=True))

        #print('p2e_feat[3]:',p2e_feat[3].size())
        #print('p2e_feat[4]:',p2e_feat[4].size())
        p2efea1 = self.convtp1(p2e_feat[2])
        #print('p2efea1:', p2efea1.size())
        p2e_feat3 = F.interpolate(p2e_feat[3], size=(64, 128), mode='bilinear', align_corners=True)
        p2efea2 = torch.cat([p2efea1, p2e_feat3], 1)
        p2efea2 = self.convtp2(p2efea2)
        p2e_feat4 = F.interpolate(p2e_feat[4], size=(64, 128), mode='bilinear', align_corners=True)
        p2efea3 = torch.cat([p2efea2, p2e_feat4], 1)
        p2efea3 = self.convtp3(p2efea3)

        pred_tp = self.convtp_f(p2efea3)
        pred_tp = torch.sigmoid(F.interpolate(pred_tp,
                                               size=(H_e, W_e),
                                               mode='bilinear',
                                               align_corners=True))

        sat_uncer_list0 = F.interpolate(sat_uncer_list0,size=(H_e, W_e),mode='bilinear', align_corners=True)
        sat_uncer_list1 = F.interpolate(sat_uncer_list1,size=(H_e, W_e),mode='bilinear', align_corners=True)
        sat_uncer_list2 = F.interpolate(sat_uncer_list2,size=(H_e, W_e),mode='bilinear', align_corners=True)

        pred_final = ((pred_erp + pred_tp)*sat_uncer_list0 + (pred_erp + pred_tp)*sat_uncer_list1 + (pred_erp + pred_tp)*sat_uncer_list2)/3

        return pred_erp, pred_tp, pred_final

    def load_pre(self, pre_model):
        trained_model_state_dict = torch.load(pre_model)
        # print(trained_model_state_dict.keys())
        model_state_dict = self.equi_encoder.state_dict()
        # print(model_state_dict.keys())
        self.equi_encoder.load_state_dict({k: v for k, v in trained_model_state_dict.items() if k in model_state_dict},
                                          strict=False)
        # self.erp_hivit.load_state_dict(torch.load(pre_model)['model'],strict=False)
        # print(f"RGB SwinTransformer loading pre_model ${pre_model}")
