# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmaction.core.bbox import bbox2result
from mmaction.utils import import_module_error_class
from mmdet.models import SHARED_HEADS

from torch import nn as nn

try:
    from mmdet.core.bbox import bbox2roi
    from mmdet.models.roi_heads import StandardRoIHead
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False

if mmdet_imported:

    @SHARED_HEADS.register_module()
    class IdentitySharedHead(nn.Module):
        def __init__(self):
            super().__init__()

        def init_weights(self, pretrained=None):
            self.pretrained = pretrained

        def forward(self, x):
            return x

    @SHARED_HEADS.register_module()
    class MLPSharedHead(nn.Module):
        def __init__(self, dim_in, dim, pretrained=None):
            super().__init__()
            self.mlp = nn.Sequential(nn.Conv3d(dim_in, dim_in, 1), nn.ReLU(), nn.Conv3d(dim_in, dim, 1))
            self.pretrained = pretrained

        def init_weights(self, pretrained=None):
            self.pretrained = pretrained

        def forward(self, x, **kwargs):
            return self.mlp(x)

else:
    # Just define an empty class, so that __init__ can import it.
    @import_module_error_class('mmdet')
    class IdentitySharedHead:
        pass
