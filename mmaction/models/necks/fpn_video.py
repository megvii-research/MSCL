from mmcv.cnn import ConvModule, constant_init, normal_init, xavier_init
from torch import nn as nn

from .fpn import FPN
from .sepc import SEPC


class TemporalModulation(nn.Module):
    """Temporal Rate Modulation.

    The module is used to equip TPN with a similar flexibility for temporal
    tempo modulation as in the input-level frame pyramid.


    Args:
        in_channels (int): Channel number of input features.
        out_channels (int): Channel number of output features.
        downsample_scale (int): Downsample scale for maxpooling. Default: 8.
    """

    def __init__(self, in_channels, out_channels, downsample_scale=8):
        super().__init__()

        self.conv = ConvModule(
            in_channels,
            out_channels,
            (3, 1, 1),
            stride=(1, 1, 1),
            padding=(1, 0, 0),
            bias=False,
            groups=32,
            conv_cfg=dict(type="Conv3d"),
            act_cfg=None,
        )
        self.pool = nn.MaxPool3d((downsample_scale, 1, 1), (downsample_scale, 1, 1), (0, 0, 0), ceil_mode=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class TPNSingle(nn.Module):
    """TPN neck.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        fpn_cfg=dict(fpn_kerne_size=(1, 3, 3), conv_cfg=dict(type="Conv3d")),
        temporal_modulation_cfg=None,
        sepc_cfg=None,
        aux_head_cfg=None,
        flow_type="top-down",
        reverse_st=False,
    ):
        super().__init__()
        assert isinstance(in_channels, list)
        assert isinstance(out_channels, int)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_tpn_stages = len(in_channels)

        assert temporal_modulation_cfg is None or isinstance(temporal_modulation_cfg, dict)
        assert aux_head_cfg is None or isinstance(aux_head_cfg, dict)

        if flow_type not in ["top-down"]:
            raise ValueError(f"flow type in TPN should be 'top-down'" f"but got {flow_type} instead.")
        self.flow_type = flow_type
        self.reverse_st = reverse_st

        self.temporal_modulation_ops = nn.ModuleList()
        self.fpn_upsample_ops = nn.ModuleList()

        # * This module only consider relation in spatial, so tepmoral kernel size>1 is not needed.
        self.fpn = FPN(in_channels, out_channels, self.num_tpn_stages, **fpn_cfg)

        for i in range(self.num_tpn_stages):
            cur_channels = in_channels[i] if self.reverse_st else out_channels
            if temporal_modulation_cfg is not None:
                downsample_scale = temporal_modulation_cfg["downsample_scales"][i]
                temporal_modulation = TemporalModulation(cur_channels, cur_channels, downsample_scale)
                self.temporal_modulation_ops.append(temporal_modulation)

        if sepc_cfg is not None:
            self.sepc = SEPC(**sepc_cfg)
        else:
            self.sepc = None

        # out_dims = level_fusion_cfg['out_channels']
        self.aux_head = None

        self.init_weights()

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                xavier_init(m, distribution="uniform")
            if isinstance(m, nn.BatchNorm3d):
                constant_init(m, 1)

        self.fpn.init_weights()
        if self.sepc is not None:
            self.sepc.init_weights()
        if self.aux_head is not None:
            self.aux_head.init_weights()

    def forward(self, x):
        # loss_aux = dict()

        # # Auxiliary loss
        # if self.aux_head is not None:
        #     loss_aux = self.aux_head(x[-2], target)
        x = x[-self.num_tpn_stages :]

        if self.reverse_st:
            outs = x
            for i, temporal_modulation in enumerate(self.temporal_modulation_ops):
                outs[i] = temporal_modulation(outs[i])
            outs = self.fpn(outs)
        else:
            # Spatial Modulation
            outs = self.fpn(x)

            # FPN Conv + Temporal Modulation
            for i, temporal_modulation in enumerate(self.temporal_modulation_ops):
                outs[i] = temporal_modulation(outs[i])

        # SEPC
        # * SEPC, for scale equivalence feature
        if self.sepc:
            outs = self.sepc(outs)

        return outs