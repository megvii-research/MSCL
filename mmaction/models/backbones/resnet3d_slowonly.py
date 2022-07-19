# Copyright (c) OpenMMLab. All rights reserved.
from turtle import forward
from ..builder import BACKBONES
from .resnet3d_slowfast import ResNet3dPathway

import copy

try:
    from mmdet.models.builder import BACKBONES as MMDET_BACKBONES
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False


@BACKBONES.register_module()
class ResNet3dSlowOnly(ResNet3dPathway):
    """SlowOnly backbone based on ResNet3dPathway.

    Args:
        *args (arguments): Arguments same as :class:`ResNet3dPathway`.
        conv1_kernel (Sequence[int]): Kernel size of the first conv layer.
            Default: (1, 7, 7).
        conv1_stride_t (int): Temporal stride of the first conv layer.
            Default: 1.
        pool1_stride_t (int): Temporal stride of the first pooling layer.
            Default: 1.
        inflate (Sequence[int]): Inflate Dims of each block.
            Default: (0, 0, 1, 1).
        **kwargs (keyword arguments): Keywords arguments for
            :class:`ResNet3dPathway`.
    """

    def __init__(self,
                 *args,
                 lateral=False,
                 conv1_kernel=(1, 7, 7),
                 conv1_stride_t=1,
                 pool1_stride_t=1,
                 inflate=(0, 0, 1, 1),
                 with_pool2=False,
                 **kwargs):
        super().__init__(
            *args,
            lateral=lateral,
            conv1_kernel=conv1_kernel,
            conv1_stride_t=conv1_stride_t,
            pool1_stride_t=pool1_stride_t,
            inflate=inflate,
            with_pool2=with_pool2,
            **kwargs)

        assert not self.lateral


@BACKBONES.register_module()
class ResNet3dSlowOnly_TwoR5(ResNet3dPathway):
    """SlowOnly backbone based on ResNet3dPathway.

    Args:
        *args (arguments): Arguments same as :class:`ResNet3dPathway`.
        conv1_kernel (Sequence[int]): Kernel size of the first conv layer.
            Default: (1, 7, 7).
        conv1_stride_t (int): Temporal stride of the first conv layer.
            Default: 1.
        pool1_stride_t (int): Temporal stride of the first pooling layer.
            Default: 1.
        inflate (Sequence[int]): Inflate Dims of each block.
            Default: (0, 0, 1, 1).
        **kwargs (keyword arguments): Keywords arguments for
            :class:`ResNet3dPathway`.
    """

    def __init__(self,
                 *args,
                 lateral=False,
                 conv1_kernel=(1, 7, 7),
                 conv1_stride_t=1,
                 pool1_stride_t=1,
                 inflate=(0, 0, 1, 1),
                 with_pool2=False,
                 **kwargs):
        super().__init__(
            *args,
            lateral=lateral,
            conv1_kernel=conv1_kernel,
            conv1_stride_t=conv1_stride_t,
            pool1_stride_t=pool1_stride_t,
            inflate=inflate,
            with_pool2=with_pool2,
            **kwargs)

        assert not self.lateral
        ori_layer_name = self.res_layers[-1]
        ori_layer = getattr(self, ori_layer_name)

        new_layer_name = ori_layer_name + '_local'
        setattr(self, new_layer_name, copy.deepcopy(ori_layer))
        self.res_layers[-1] = (ori_layer_name, new_layer_name)
        print(self.res_layers)

    def forward(self, x):
        x = self.conv1(x)
        if self.with_pool1:
            x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers[:-1]):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i == 0 and self.with_pool2:
                x = self.pool2(x)
            if i in self.out_indices:
                outs.append(x)

        global_res_layer = getattr(self, self.res_layers[-1][0])
        local_res_layer = getattr(self, self.res_layers[-1][1])
        x_g = global_res_layer(x)
        x_l = local_res_layer(x)
        if (len(self.res_layers)-1) in self.out_indices:
            outs.append((x_g, x_l))
        if len(outs) == 1:
            return outs[0]

        return tuple(outs)

if mmdet_imported:
    MMDET_BACKBONES.register_module()(ResNet3dSlowOnly)