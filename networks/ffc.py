import torch
import torch.nn as nn
from .non_local import NONLocalBlock2D

try:
    from torch import irfft
    from torch import rfft
except ImportError:
    from torch.fft import irfft2
    from torch.fft import rfft2
    def rfft(x, d):
        t = rfft2(x, dim = (-d,-1))
        return torch.stack((t.real, t.imag), -1)
    def irfft(x, d, signal_sizes):
        return irfft2(torch.complex(x[:,:,0], x[:,:,1]), s = signal_sizes, dim = (-d,-1))

# class Non_Local_Attention_Block(nn.Module):
#     def __init__(self, channels):
#         super(Non_Local_Attention_Block, self).__init__()
    
#     def forward(self, x):
#         return 0

class FFCSE_block(nn.Module):

    def __init__(self, channels, ratio_g):
        super(FFCSE_block, self).__init__()
        in_cg = int(channels * ratio_g)
        in_cl = channels - in_cg
        r = 16

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(channels, channels // r,
                               kernel_size=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_a2l = None if in_cl == 0 else nn.Conv2d(
            channels // r, in_cl, kernel_size=1, bias=True)
        self.conv_a2g = None if in_cg == 0 else nn.Conv2d(
            channels // r, in_cg, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x

        x = id_l if type(id_g) is int else torch.cat([id_l, id_g], dim=1)
        x = self.avgpool(x)
        x = self.relu1(self.conv1(x))

        x_l = 0 if self.conv_a2l is None else id_l * \
            self.sigmoid(self.conv_a2l(x))
        x_g = 0 if self.conv_a2g is None else id_g * \
            self.sigmoid(self.conv_a2g(x))
        return x_l, x_g


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()
        r_size = x.size()

        # (batch, c, h, w/2+1, 2)
        ffted = rfft(x, 2)
        # (batch, c, 2, h, w/2+1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)

        output = irfft(ffted, 2, signal_sizes=r_size[2:])

        return output


class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s_h = h // split_no
            split_s_w = w // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s_h, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s_w, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output


class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 ratio_gin=0.5, ratio_gout=0.5, stride=1, padding=1,
                 dilation=1, groups=1, bias=False, enable_lfu=True, norm_layer=nn.BatchNorm2d):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        self.in_cg = int(in_channels * ratio_gin)
        self.in_cl = in_channels - self.in_cg
        self.out_cg = int(out_channels * ratio_gout)
        self.out_cl = out_channels - self.out_cg

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        # local -> local
        module = nn.Identity if self.in_cl == 0 or self.out_cl == 0 else nn.Conv2d
        self.convl2l = nn.Sequential(
            module(self.in_cl, self.out_cl, kernel_size,stride, padding, dilation, groups, bias),
            nn.LeakyReLU()
        )
        
        # local -> gloabl   ============ todo: add the Non-Local Attention
        module = nn.Identity if self.in_cl == 0 or self.out_cg == 0 else nn.Conv2d
        self.convl2g = nn.Sequential(
            module(self.in_cl, self.out_cg, kernel_size, stride, padding, dilation, groups, bias),
            NONLocalBlock2D(self.out_cg)
        )
        
        # gloabl -> local
        module = nn.Identity if self.in_cg == 0 or self.out_cl == 0 else nn.Conv2d
        self.convg2l = module(self.in_cg, self.out_cl, kernel_size,
                              stride, padding, dilation, groups, bias)
        
        # gloabl -> gloabl
        module = nn.Identity if self.in_cg == 0 or self.out_cg == 0 else SpectralTransform
        self.convg2g = nn.Sequential(
            module(self.in_cg, self.out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu),
            nn.LeakyReLU() 
        )
        
        self.local_bn_relu = (
            nn.Sequential(norm_layer(self.out_cl), nn.ReLU(True))
            if self.out_cl != 0
            else nn.Identity()
        )

        self.global_bn_relu = (
            nn.Sequential(norm_layer(self.out_cg), nn.ReLU(True))
            if self.out_cg != 0
            else nn.Identity()
        )

    def forward(self, x):
        x_l, x_g = (
            x[:, : self.in_cl, ...],
            x[:, self.in_cl :, ...],
        )
        x_l = 0 if x_l.size()[1] == 0 else x_l
        x_g = 0 if x_g.size()[1] == 0 else x_g
        # x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)
            out_xl = self.local_bn_relu(out_xl)
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) + self.convg2g(x_g)
            out_xg = self.global_bn_relu(out_xg)
         #  (B, out_ch, F, T)
        output = torch.cat((out_xl, out_xg), dim=1)
        return output

class NL_FFC(torch.nn.Module):
    """Implements Residual FFC block.

    Contains two FFC blocks with residual connection.

    Wraps around FFC arguments.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.ffc1 = FFC(in_channels, out_channels)
        self.ffc2 = FFC(in_channels, out_channels)

    def forward(self, x):
        out = self.ffc1(x)
        out = self.ffc2(out)
        
        return x + out
    
if __name__ == '__main__':
    x       = torch.randn(1,64,32,32)
    ffc   = NL_FFC(64, 64)
    print(ffc(x).shape)