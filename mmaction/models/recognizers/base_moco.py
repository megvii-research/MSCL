import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseRecognizer
from .. import builder


class BaseMoCoRecognizer(BaseRecognizer):
    """Base class for MoCo recognizers.

    This is the base class for MoCo, which is adjust from BaseRecognizer in mmaction2:
    Support _build_backbone/neck/cls_head for multiple times, which is used for contrastive learning.

    Args:
        backbone (dict): Backbone modules to extract feature.
        cls_head (dict | None): Classification head to process feature.
            Default: None.
        neck (dict | None): Neck for feature fusion. Default: None.
        train_cfg (dict | None): Config for training. Default: None.
        test_cfg (dict | None): Config for testing. Default: None.
    """

    def __init__(self,
                 backbone=None,
                 cls_head=None,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None):
        super(BaseRecognizer, self).__init__()
        # record the source of the backbone
        self.backbone_from = 'mmaction2'

        self.backbone_list = []
        self.neck_list = []
        self.cls_head_list = []

        if backbone is not None:
            self._build_backbone(backbone)
        if neck is not None:
            self._build_neck(neck)
        if cls_head is not None:
            self._build_cls_head(cls_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # aux_info is the list of tensor names beyond 'imgs' and 'label' which
        # will be used in train_step and val_step, data_batch should contain
        # these tensors
        self.aux_info = []
        if train_cfg is not None and 'aux_info' in train_cfg:
            self.aux_info = train_cfg['aux_info']
        # max_testing_views should be int
        self.max_testing_views = None
        if test_cfg is not None and 'max_testing_views' in test_cfg:
            self.max_testing_views = test_cfg['max_testing_views']
            assert isinstance(self.max_testing_views, int)

        if test_cfg is not None and 'feature_extraction' in test_cfg:
            self.feature_extraction = test_cfg['feature_extraction']
        else:
            self.feature_extraction = False

        # mini-batch blending, e.g. mixup, cutmix, etc.
        self.blending = None
        if train_cfg is not None and 'blending' in train_cfg:
            from mmcv.utils import build_from_cfg
            from mmaction.datasets.builder import BLENDINGS
            self.blending = build_from_cfg(train_cfg['blending'], BLENDINGS)

        self.init_weights()

        self.fp16_enabled = False
    
    def _build_backbone(self, backbone, name="backbone"):
        # record the source of the backbone

        # TODO: support mmcls build like mmaction2.
        backbone = backbone.copy()
        if backbone["type"].startswith("torchvision."):
            try:
                # import torchvision.models
                from torchvision.models import video as models
            except (ImportError, ModuleNotFoundError):
                raise ImportError("Please install torchvision to use this " "backbone.")
            backbone_type = backbone.pop("type")[12:]
            backbone = models.__dict__[backbone_type](**backbone)
            # disable the classifier
            backbone.classifier = nn.Identity()
            backbone.fc = nn.Identity()
            self.backbone_from = "torchvision"
            setattr(self, name, backbone)
        elif backbone["type"].startswith("resnet_flow."):
            from ..backbones.fastonly import ResNetFlow
            backbone_type = backbone.pop("type")[12:]
            backbone = ResNetFlow(backbone_type, **backbone)
            # disable the classifier
            backbone.classifier = nn.Identity()
            backbone.fc = nn.Identity()
            self.backbone_from = "torchvision"
            setattr(self, name, backbone)
        else:
            setattr(self, name, builder.build_backbone(backbone))
        self.backbone_list.append(name)

    def _build_neck(self, neck, name="neck"):
        setattr(self, name, builder.build_neck(neck))
        self.neck_list.append(name)

    def _build_cls_head(self, cls_head, name="cls_head"):
        setattr(self, name, builder.build_head(cls_head))
        self.cls_head_list.append(name)

    @property
    def with_neck(self):
        """bool: whether the recognizer has a neck"""
        return len(self.neck_list) > 0

    @property
    def with_cls_head(self):
        """bool: whether the recognizer has a cls_head"""
        return len(self.cls_head_list) > 0

    def init_weights(self):
        """Initialize the model network weights."""
        done = []
        for bn in self.backbone_list:
            if self.backbone_from in ["mmcls", "megaction", "mmaction2"]:
                getattr(self, bn).init_weights()
            elif self.backbone_from == "torchvision":
                warnings.warn(
                    "We do not initialize weights for backbones in "
                    "torchvision, since the weights for backbones in "
                    "torchvision are initialized in their __init__ "
                    "functions. "
                )
            else:
                raise NotImplementedError("Unsupported backbone source " f"{self.backbone_from}!")
            done.append(bn)

        for nname in self.neck_list:
            getattr(self, nname).init_weights()
            done.append(nname)
        for cn in self.cls_head_list:
            getattr(self, cn).init_weights()
            done.append(cn)
        print(f"Init weights for {done} !!")

    def forward_train(self, imgs, labels, **kwargs):
        """Defines the computation performed at every call when training."""
        raise NotImplementedError("Not support forward_train for BaseMoCoRecognizer")

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        raise NotImplementedError("Not support forward_test for BaseMoCoRecognizer")

    def forward_gradcam(self, imgs):
        """Defines the computation performed at every all when using gradcam
        utils."""
        raise NotImplementedError("Not support forward_gradcam for BaseMoCoRecognizer")