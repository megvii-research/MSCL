# Copyright (c) OpenMMLab. All rights reserved.
from .audio_tsn_head import AudioTSNHead
from .base import BaseHead
from .bbox_head import BBoxHeadAVA
from .fbo_head import FBOHead
from .i3d_head import I3DHead
from .lfb_infer_head import LFBInferHead
from .misc_head import ACRNHead
from .roi_head import AVARoIHead
from .slowfast_head import SlowFastHead
from .ssn_head import SSNHead
from .timesformer_head import TimeSformerHead
from .tpn_head import TPNHead
from .trn_head import TRNHead
from .tsm_head import TSMHead
from .tsn_head import TSNHead
from .x3d_head import X3DHead
from .reid_head import TSMReidSimpleHead
from .moco_head import MoCoHead, MoDistPredHead, FGMoDistPredHead, MTMoDistPredHead, MoDistMSEPredHead
from .ssl_roi_head import SSLRoIHead
from .shared_head import IdentitySharedHead, MLPSharedHead
from .distill_head import RcMoDistHead
from .moco_head_v2 import MSCLWithAugMxHead, MSCLWithAugPosHead, MSCLWithAugSimpleHead, MAMSCLWithAugPosHead, MlvlMSCLWithAugPosHead, MoDistv2PosHead
from .moco_head_v3 import MoCoHeadV2, MSCLWithAugDistillMxHead, MSCLWithAugMSFMxHead, MSFHead

__all__ = [
    'TSNHead', 'I3DHead', 'BaseHead', 'TSMHead', 'SlowFastHead', 'SSNHead',
    'TPNHead', 'AudioTSNHead', 'X3DHead', 'BBoxHeadAVA', 'AVARoIHead',
    'FBOHead', 'LFBInferHead', 'TRNHead', 'TimeSformerHead', 'ACRNHead'
]
