"""Deformable ConvNets v2 in PyTorch."""

import math
from typing import Tuple, Union

import torch
import torchvision.ops
from torch import nn
from torch.nn.modules.utils import _pair

import sys
sys.path.append('../')
from utils import equi2panels

# copy from https://github.com/liyier90/pytorch-dcnv2/tree/master

class DCNv2(nn.Module):
    num_chunks = 3  # Num channels for offset + mask

    def __init__(  # pylint: disable=too-many-arguments
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]],
        dilation: Union[int, Tuple[int, int]] = 1,
        deformable_groups: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

        num_offset_mask_channels = (
            self.deformable_groups
            * self.num_chunks
            * self.kernel_size[0]
            * self.kernel_size[1]
        )
        # print(num_offset_mask_channels)
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            num_offset_mask_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            bias=True,
        )
        self.init_offset()

    def forward(self, input: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        out = self.conv_offset_mask(offsets)
        offset_1, offset_2, mask = torch.chunk(out, self.num_chunks, dim=1)
        offset = torch.cat((offset_1, offset_2), dim=1).float()
        mask = torch.sigmoid(mask).float()
        # print(offset.dtype,mask.dtype)
        return torchvision.ops.deform_conv2d(
            input=input,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )

    def init_offset(self) -> None:
        """Initializes the weight and bias for `conv_offset_mask`."""
        self.conv_offset_mask.weight.data.zero_()
        if self.conv_offset_mask.bias is not None:
            self.conv_offset_mask.bias.data.zero_()

    def reset_parameters(self) -> None:
        """Re-initialize parameters using a method similar to He
        initialization with mode='fan_in' and gain=1.
        """
        fan_in = self.in_channels
        for k in self.kernel_size:
            fan_in *= k
        std = 1.0 / math.sqrt(fan_in)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()


def get_condition(h, w):
    
    return torch.cos(make_coord([h]).unsqueeze(1).repeat([1, w, 1]).permute(2,0,1) * math.pi / 2)
 

def make_coord(shape, ranges=(-1, 1), flatten=False):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        v0, v1 = ranges
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

if __name__ == '__main__':
    
    condition_dim = 1
    c_dim = 128
    dim = 128
    B = 2
    I_rad, S_rad, N = math.pi * 0.5, math.pi * 0.25, 8
    
    # todo: get offsets
    offset_conv = nn.Sequential(nn.Conv2d(condition_dim, c_dim, 1, 1, 0, bias=True),
                                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                nn.Conv2d(c_dim, dim, 1, 1, 0, bias=True),
                                nn.LeakyReLU(negative_slope=0.2, inplace=True)).cuda()
    condition = get_condition(512, 1024).cuda() # [1, 512, 1024]
    condition = condition.unsqueeze(0).expand(B, -1, -1, -1) # [B, 1, 512, 1024]
    condition_e2p, _ = equi2panels(condition, I_rad, S_rad, N) # # [B*N, 1, 512, 256]
    offset = offset_conv(condition_e2p) # [B*N, dim, 512, 256]
    # wrap all things (offset and mask) in DCN
    x = torch.randn(B, dim, 512, 1024).cuda()
    dcn = DCNv2(dim, dim, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2).cuda()
    for name, param in dcn.named_parameters():
        print(f"Parameter name: {name}, Data type: {param.dtype}")
    # print(x.dtype, offset.dtype. dcn.dtype)

    output = dcn(x, offset)
    # print(output.shape)