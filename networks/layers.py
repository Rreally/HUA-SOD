import torch
from torch import nn
from torch.nn import functional as F

from .dcn import DCNv2, get_condition
from .ffc import NL_FFC

import sys
sys.path.append('../')
from utils import equi2pers

# BAM
class BAM(nn.Module):
    def __init__(self, in_c):
        super(BAM, self).__init__()
        self.reduce = nn.Conv2d(in_c, 32, 1)
        self.ff_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.bf_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.offset_conv = nn.Sequential(
            nn.Conv2d(1, 32, 1, 1, 0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, 32, 1, 1, 0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        
        self.enhance_conv = nn.Sequential(
            nn.Conv2d(32*2, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.dcn = DCNv2(32, 32, kernel_size=(3, 3), 
                         stride=1, padding=1, deformable_groups=2)
        self.rgbd_pred_layer = Pred_Layer(32)

    def get_offset(self, B, H, W): 
        condition = get_condition(H, W).cuda()
        condition = condition.unsqueeze(0).expand(B, -1, -1, -1)
        offset = self.offset_conv(condition).float()
        return offset
    
    def forward(self, rgb_feat, pred, uncertain=None):
        feat = self.reduce(rgb_feat)
        [B, _, H, W] = rgb_feat.size()
        
        pred = torch.sigmoid(
            F.interpolate(pred,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True))
        ff_feat = self.ff_conv(feat * pred)
        bf_feat = self.bf_conv(feat * (1 - pred))
        offset = self.get_offset(B, H, W)
        enhance_feat = self.enhance_conv(torch.cat((ff_feat, bf_feat), 1))
        enhance_feat = self.dcn(enhance_feat, offset)
        
        new_pred = self.rgbd_pred_layer(enhance_feat)
        return new_pred

# MBAM
class MBAM(nn.Module):
    def __init__(self, in_c):
        super(MBAM, self).__init__()
        self.ff_conv = ASPP(in_c)
        self.bf_conv = ASPP(in_c)
        self.offset_conv = nn.Sequential(
            nn.Conv2d(1, 32, 1, 1, 0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, 32, 1, 1, 0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        
        self.enhance_conv = nn.Sequential(
            nn.Conv2d(32*8, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.dcn = DCNv2(32, 32, kernel_size=(3, 3), 
                         stride=1, padding=1, deformable_groups=2)
        self.rgbd_pred_layer = Pred_Layer(32)
        
    def get_offset(self, B, H, W): 
        condition = get_condition(H, W).cuda()
        condition = condition.unsqueeze(0).expand(B, -1, -1, -1)
        offset = self.offset_conv(condition).float()
        return offset
    
    def forward(self, feat, pred, uncertain=None):
        
        [B, _, H, W] = feat.size()
        pred = torch.sigmoid(
            F.interpolate(pred,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True))
        # print(uncertain.shape, pred.shape)
        pred = pred + uncertain
        ff_feat = self.ff_conv(feat * pred)
        bf_feat = self.bf_conv(feat * (1 - pred))
        offset = self.get_offset(B, H, W)
        enhance_feat = self.enhance_conv(torch.cat((ff_feat, bf_feat), 1))
        enhance_feat = self.dcn(enhance_feat, offset)
        new_pred = self.rgbd_pred_layer(enhance_feat)
        return new_pred

class Att(nn.Module):
    def __init__(self, dim):
        super(Att, self).__init__()
        self.dcn = DCNv2(dim, dim, kernel_size=(3, 3), 
                         stride=1, padding=1, deformable_groups=2)
        self.offset_conv = nn.Sequential(
            nn.Conv2d(1, dim, 1, 1, 0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.scconv = ScConv(dim)
    
    def get_offset(self, Hp, Wp, B, k): 
        """
            Hp: pers size
            Wp: pers size
            B : ERP图像输入的batchsize
            k : 与输入图像的缩放比例
        """
        H_c = 512 // k
        W_c = 1024 // k
        condition = get_condition(H_c, W_c).cuda()
        condition = condition.unsqueeze(0).expand(B, -1, -1, -1)
        # condition_e2p, _, _, _ = equi2pers(condition, nrows=4, fov=(80, 80), patch_size=(Hp, Wp))
        condition_e2p, _, _, _ = equi2pers(condition, nrows=3, fov=(120, 120), patch_size=(Hp, Wp)) # B, C, H, W, N
        
        offset = self.offset_conv(condition_e2p).float()
        
        return offset
    
    def forward(self, pers, B):
        Bp, C, Hp, Wp = pers.shape
        k = 224 // Hp
        offset = self.get_offset(Hp, Wp, B, k)
        x_att = self.dcn(pers, offset)
        x_res = self.scconv(pers)
        x_att = x_att + x_res
        return x_att


class Pred_Layer(nn.Module):
    def __init__(self, in_c=32):
        super(Pred_Layer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_c, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.outlayer = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0), )

    def forward(self, x):
        x = self.enlayer(x)
        x = self.outlayer(x)
        return x
    
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, dilation=dilation, bias=False)
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride, dilation, dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out

# ASPP for MBAM
class ASPP(nn.Module):
    def __init__(self, in_c):
        super(ASPP, self).__init__()

        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_c, 32, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_c, 32, 3, 1, padding=3, dilation=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_c, 32, 3, 1, padding=5, dilation=5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(in_c, 32, 3, 1, padding=7, dilation=7),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x

  
class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, bias=True):
        super(Conv3x3, self).__init__()

        self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3, bias=bias)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out
    
class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels, bias=True):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels, bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(self.bn(out))
        return out
    
class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num:int, 
                 group_num:int = 16, 
                 eps:float = 1e-10
                 ):
        super(GroupBatchnorm2d,self).__init__()
        assert c_num    >= group_num
        self.group_num  = group_num
        self.weight     = nn.Parameter( torch.randn(c_num, 1, 1)    )
        self.bias       = nn.Parameter( torch.zeros(c_num, 1, 1)    )
        self.eps        = eps
    def forward(self, x):
        N, C, H, W  = x.size()
        x           = x.view(   N, self.group_num, -1   )
        mean        = x.mean(   dim = 2, keepdim = True )
        std         = x.std (   dim = 2, keepdim = True )
        x           = (x - mean) / (std+self.eps)
        x           = x.view(N, C, H, W)
        return x * self.weight + self.bias


class SRU(nn.Module):
    def __init__(self,
                 oup_channels:int, 
                 group_num:int = 16,
                 gate_treshold:float = 0.5,
                 torch_gn:bool = True
                 ):
        super().__init__()
        
        self.gn             = nn.GroupNorm( num_channels = oup_channels, num_groups = group_num ) if torch_gn else GroupBatchnorm2d(c_num = oup_channels, group_num = group_num)
        self.gate_treshold  = gate_treshold
        self.sigomid        = nn.Sigmoid()

    def forward(self,x):
        gn_x        = self.gn(x)
        w_gamma     = self.gn.weight/sum(self.gn.weight)
        w_gamma     = w_gamma.view(1,-1,1,1)
        reweigts    = self.sigomid( gn_x * w_gamma )
        # Gate
        w1          = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts) # 大于门限值的设为1，否则保留原值
        w2          = torch.where(reweigts > self.gate_treshold, torch.zeros_like(reweigts), reweigts) # 大于门限值的设为0，否则保留原值
        x_1         = w1 * x
        x_2         = w2 * x
        y           = self.reconstruct(x_1,x_2)
        return y
    
    def reconstruct(self,x_1,x_2):
        x_11,x_12 = torch.split(x_1, x_1.size(1)//2, dim=1)
        x_21,x_22 = torch.split(x_2, x_2.size(1)//2, dim=1)
        return torch.cat([ x_11+x_22, x_12+x_21 ],dim=1)


class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''
    def __init__(self, 
                 op_channel:int,
                 alpha:float = 1/2,
                 squeeze_radio:int = 2 ,
                 group_size:int = 2,
                 group_kernel_size:int = 3,
                 ):
        super().__init__()
        self.up_channel     = up_channel   =   int(alpha*op_channel)
        self.low_channel    = low_channel  =   op_channel-up_channel
        self.squeeze1       = nn.Conv2d(up_channel,up_channel//squeeze_radio,kernel_size=1,bias=False)
        self.squeeze2       = nn.Conv2d(low_channel,low_channel//squeeze_radio,kernel_size=1,bias=False)
        #up
        self.GWC            = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=group_kernel_size, stride=1,padding=group_kernel_size//2, groups = group_size)
        self.PWC1           = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=1, bias=False)
        #low
        self.PWC2           = nn.Conv2d(low_channel//squeeze_radio, op_channel-low_channel//squeeze_radio,kernel_size=1, bias=False)
        self.advavg         = nn.AdaptiveAvgPool2d(1)

    def forward(self,x):
        # Split
        up,low  = torch.split(x,[self.up_channel,self.low_channel],dim=1)
        up,low  = self.squeeze1(up),self.squeeze2(low)
        # Transform
        Y1      = self.GWC(up) + self.PWC1(up)
        Y2      = torch.cat( [self.PWC2(low), low], dim= 1 )
        # Fuse
        out     = torch.cat( [Y1,Y2], dim= 1 )
        out     = F.softmax( self.advavg(out), dim=1 ) * out
        out1,out2 = torch.split(out,out.size(1)//2,dim=1)
        return out1+out2


class ScConv(nn.Module):
    def __init__(self,
                op_channel:int,
                group_num:int = 4,
                gate_treshold:float = 0.5,
                alpha:float = 1/2,
                squeeze_radio:int = 2 ,
                group_size:int = 2,
                group_kernel_size:int = 3,
                 ):
        super().__init__()
        self.SRU = SRU( op_channel, 
                       group_num            = group_num,  
                       gate_treshold        = gate_treshold )
        self.CRU = CRU( op_channel, 
                       alpha                = alpha, 
                       squeeze_radio        = squeeze_radio ,
                       group_size           = group_size ,
                       group_kernel_size    = group_kernel_size )
    
    def forward(self,x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x

class FeatureProcessBlock(nn.Module):
    # baseffc版本
    def __init__(self, in_channels, out_channels, groups=1):
        # bn_layer not used
        super(FeatureProcessBlock, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2, kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()
        xx = x.clone()

        # (batch, c, h, w/2+1, 2)
        ffted = torch.fft.rfft2(xx, norm = 'ortho')
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])  # (batch, c*2, h, w/2+1)
        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        output = torch.fft.irfft2(ffted, norm='ortho')
        output = output.view(batch, c, h, w).contiguous()

        return output

# from https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class FFM(nn.Module):
    def __init__(self, in_c, out_c):
        super(FFM, self).__init__()
        self.sea = SELayer(out_c)
        self.equi_enhance_conv = nn.Sequential(
            nn.Conv2d(out_c * 2, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.p2e_enhance_conv = nn.Sequential(
            nn.Conv2d(out_c * 2, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(out_c * 2, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.reduce_conv = nn.Conv2d(in_c, out_c, 1, bias=False)

    def forward(self, equi_feat, p2e_feat):
        [B, C, H, W] = p2e_feat.size()
        p2e_feat = self.reduce_conv(p2e_feat)
        ep_feature = self.sea(equi_feat * p2e_feat)
        equi_enhance = self.equi_enhance_conv(torch.cat([equi_feat, ep_feature], 1))
        p2e_enhance = self.p2e_enhance_conv(torch.cat([p2e_feat, ep_feature], 1))
        out = self.fuse_conv(torch.cat([equi_enhance, p2e_enhance], 1))

        return out
    
if __name__ == '__main__':
    x       = torch.randn(1,32,16,16)
    model   = ScConv(32)
    print(model(x).shape)