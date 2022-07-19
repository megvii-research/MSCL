from functools import reduce

import torch
from mmcv.runner import auto_fp16
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init

from ..builder import NECKS

sepc_conv = nn.Conv3d
# spec_conv support deform_conv, with start_level
# now, we replace it by simple nn.conv


@NECKS.register_module
class SEPC(nn.Module):
    def __init__(
        self,
        in_channels=[256] * 3,
        out_channels=256,
        stride=(2, 1, 1),
        # pconv_deform=False,
        # lcconv_deform=False,
        iBN=False,
        Pconv_num=2,
    ):
        super(SEPC, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.fp16_enabled = False
        self.iBN = iBN
        self.Pconvs = nn.ModuleList()

        for i in range(Pconv_num):
            # This follow original code, but this may cannot fix in_channels != out_channels
            self.Pconvs.append(PConv3D(in_channels[i], out_channels, stride, iBN=self.iBN,))

        self.init_weights()

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        pass  # PConv3D has its own init function!

    @auto_fp16()
    def forward(self, x):
        assert len(x) == len(self.in_channels)
        x = x
        for pconv in self.Pconvs:
            x = pconv(x)

        return x


class PConv3D(nn.Module):
    def __init__(
        self,
        in_channels=256,
        out_channels=256,
        stride=(2, 1, 1),  # stride between two layers.
        kernel_size=[3, 3, 3],
        dilation=[1, 1, 1],
        groups=[1, 1, 1],
        iBN=False,
        # part_deform=False,
    ):
        super().__init__()

        #     assert not (bias and iBN)
        self.iBN = iBN
        self.Pconv = nn.ModuleList()
        self.Pconv.append(
            sepc_conv(
                in_channels,
                out_channels,
                kernel_size=kernel_size[0],
                dilation=dilation[0],
                groups=groups[0],
                padding=(kernel_size[0] + (dilation[0] - 1) * 2) // 2,
            )
        )
        self.Pconv.append(
            sepc_conv(
                in_channels,
                out_channels,
                kernel_size=kernel_size[1],
                dilation=dilation[1],
                groups=groups[1],
                padding=(kernel_size[1] + (dilation[1] - 1) * 2) // 2,
            )
        )
        self.Pconv.append(
            sepc_conv(
                in_channels,
                out_channels,
                kernel_size=kernel_size[2],
                dilation=dilation[2],
                groups=groups[2],
                padding=(kernel_size[2] + (dilation[2] - 1) * 2) // 2,
                stride=stride,
            )
        )

        if self.iBN:
            self.bn = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        for m in self.Pconv:
            init.normal_(m.weight.data, 0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        next_x = []
        for level, feature in enumerate(x):

            temp_fea = self.Pconv[1](feature)
            if level > 0:
                temp_fea += self.Pconv[2](x[level - 1])
            if level < len(x) - 1:
                temp_fea += F.interpolate(
                    self.Pconv[0](x[level + 1]),
                    size=[temp_fea.size(2), temp_fea.size(3), temp_fea.size(4)],
                    mode="trilinear",
                )
            next_x.append(temp_fea)
        if self.iBN:
            next_x = iBN(next_x, self.bn)
        next_x = [self.relu(item) for item in next_x]
        return next_x


def iBN(fms, bn):
    def mul_op(x, y):
        return x * y

    sizes = [p.shape[2:] for p in fms]
    n, c = fms[0].shape[0], fms[0].shape[1]
    fm = torch.cat([p.view(n, c, -1) for p in fms], dim=-1)
    fm = bn(fm)
    # mul_op = lambda x, y: x * y
    fm = torch.split(fm, [reduce(mul_op, s) for s in sizes], dim=-1)
    return [p.view(n, c, *s) for p, s in zip(fm, sizes)]
