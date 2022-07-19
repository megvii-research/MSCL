# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from ..builder import NECKS
from .fpn_video import TPNSingle


@NECKS.register_module()
class BaseMoCo(nn.Module):
    """
    MoCo head for pooling and generate proper outputs
    """

    def __init__(self,):
        super().__init__()
        self.tofc = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten(1))

    def forward(self, x):
        x_emb = self.tofc(x[-1])
        return (x_emb, x), dict()

    def init_weights(self):
        pass


@NECKS.register_module()
class MixBaseMoCo(nn.Module):
    """
    Mix head for BaseMoCo which append embeddings to x
    """

    def __init__(self,):
        super().__init__()
        self.tofc = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten(1))

    def forward(self, x):
        x_emb = self.tofc(x[-1])
        x.append(x_emb)
        return (x_emb, x), dict()

    def init_weights(self):
        pass


@NECKS.register_module()
class BaseMoCo_TwoR5(nn.Module):
    """
    MoCo head for pooling and generate proper outputs
    """

    def __init__(self,):
        super().__init__()
        self.tofc = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten(1))

    def forward(self, x):
        x_g, x_l = x[-1]
        x_emb = self.tofc(x_g)
        
        x_new = list(x[:-1])
        x_new.append(x_l)
        return (x_emb, x_new), dict()

    def init_weights(self):
        pass


@NECKS.register_module()
class TPNProjMoCo(nn.Module):
    def __init__(self, dims_in=(128, 256, 512), dims_out=(128, 128, 128), temporal_sizes=(4, 2, 1)):
        super().__init__()
        self.cur_rate = [temporal_sizes[0]//sz for sz in temporal_sizes]
        self.num_out_layers = len(dims_in)
        self.t_poolings = nn.ModuleList()
        self.projs = nn.ModuleList()
        for sz in temporal_sizes:
            self.t_poolings.append(nn.AdaptiveAvgPool3d((sz, None, None)))
        for i, (dim_in, dim_out) in enumerate(zip(dims_in, dims_out)):
            self.projs.append(nn.Sequential(nn.Conv3d(dim_in, dim_in//2, 1), nn.ReLU(), nn.Conv3d(dim_in//2, dim_out*self.cur_rate[i], 1)))
        self.tofc = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten(1))

    def forward(self, x: torch.Tensor):
        x_emb = self.tofc(x[-1])
        new_x = []
        for i in range(self.num_out_layers):
            cur = i - self.num_out_layers
            cur_x = self.projs[i](self.t_poolings[i](x[cur]))   # n,c,t,h,w
            num_channels = cur_x.shape[1]
            cur_x = cur_x.transpose(1, 2)   # n,t,c,h,w
            cur_x = cur_x.unflatten(2, (self.cur_rate[i], num_channels//self.cur_rate[i]))  # n,t1,t2,c,h,w
            cur_x = cur_x.flatten(1, 2).transpose(1, 2)
            new_x.append(cur_x)

        return (x_emb, new_x), dict()

    def init_weights(self):
        pass

@NECKS.register_module()
class TPNProjMoCoV2(nn.Module):
    """
    V2 -> use 50% of channels, e.g. https://arxiv.org/pdf/2007.07626.pdf
    """
    def __init__(self, dims_in=(128, 256, 512), dims_out=(128, 128, 128), ft_ids=(0, 1, 2), temporal_sizes=(4, 2, 1), chunks=(1, 2, 2)):
        super().__init__()
        self.cur_rate = [temporal_sizes[0]//sz for sz in temporal_sizes]
        self.num_out_layers = len(ft_ids)
        self.ft_ids = ft_ids
        self.chunks = chunks
        self.t_poolings = nn.ModuleList()
        self.projs = nn.ModuleList()
        for sz in temporal_sizes:
            self.t_poolings.append(nn.AdaptiveAvgPool3d((sz, None, None)))
        for i, (dim_in, dim_out, chunk) in enumerate(zip(dims_in, dims_out, chunks)):
            self.projs.append(nn.Sequential(nn.Conv3d(dim_in//chunk, dim_in//2, 1), nn.ReLU(), nn.Conv3d(dim_in//2, dim_out*self.cur_rate[i], 1)))
        self.tofc = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten(1))

    def forward(self, x: torch.Tensor):
        x_emb = self.tofc(x[-1])
        new_x = []
        for i in self.ft_ids:
            cur = i - self.num_out_layers
            cur_x = self.projs[i](self.t_poolings[i](x[cur].chunk(self.chunks[i], dim=1)[0]))   # n,c,t,h,w
            num_channels = cur_x.shape[1]
            cur_x = cur_x.transpose(1, 2)   # n,t,c,h,w
            cur_x = cur_x.unflatten(2, (self.cur_rate[i], num_channels//self.cur_rate[i]))  # n,t1,t2,c,h,w
            cur_x = cur_x.flatten(1, 2).transpose(1, 2)
            new_x.append(cur_x)

        return (x_emb, new_x), dict()

    def init_weights(self):
        pass


@NECKS.register_module()
class TPNMoCo(nn.Module):
    """
    x_emb not pass neck
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        fpn_cfg=dict(fpn_kerne_size=(1, 3, 3), conv_cfg=dict(type="Conv3d")),
        temporal_modulation_cfg=None,
        sepc_cfg=None,
        reverse_st=False,
        emb_from_bkb=True,
        # aux_head_cfg=None,
        # flow_type="top-down",
    ):
        super().__init__()
        self.tpn = TPNSingle(
            in_channels, out_channels, fpn_cfg, temporal_modulation_cfg, sepc_cfg, reverse_st=reverse_st
        )

        self.tofc = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten(1))
        self.emb_from_bkb = emb_from_bkb

        self.init_weights()

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        self.tpn.init_weights()

    def forward(self, x, target=None):
        if getattr(self, 'emb_from_bkb', True):
            x_emb = self.tofc(x[-1])
            x = self.tpn(x)
        else:
            x = self.tpn(x)
            x_emb = self.tofc(x[-1])

        return (x_emb, x), {}  # TODO: support aux head