# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import HEADS, build_loss
from .base import AvgConsensus, BaseHead


@HEADS.register_module()
class TSMReidSimpleHead(BaseHead):
    """Class head for TSM.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        num_segments (int): Number of frame segments. Default: 8.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        consensus (dict): Consensus config dict.
        dropout_ratio (float): Probability of dropout layer. Default: 0.4.
        init_std (float): Std value for Initiation. Default: 0.01.
        is_shift (bool): Indicating whether the feature is shifted.
            Default: True.
        temporal_pool (bool): Indicating whether feature is temporal pooled.
            Default: False.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_segments=8,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 loss_reid=dict(type='TripletLoss'),
                 spatial_type='avg',
                 consensus=dict(type='AvgConsensus', dim=1),
                 dropout_ratio=0.8,
                 init_std=0.001,
                 is_shift=True,
                 temporal_pool=False,
                 use_bnneck=True,
                 use_cosface=dict(use=False, s=64, m=0.1),
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.num_segments = num_segments
        self.init_std = init_std
        self.is_shift = is_shift
        self.temporal_pool = temporal_pool

        consensus_ = consensus.copy()

        consensus_type = consensus_.pop('type')
        if consensus_type == 'AvgConsensus':
            self.consensus = AvgConsensus(**consensus_)
        else:
            self.consensus = None

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.use_bnneck = use_bnneck
        if self.use_bnneck:
            self.bnneck = nn.BatchNorm1d(self.in_channels)
            self.bnneck.bias.requires_grad_(False) 

        self.use_cosface = use_cosface['use']
        if self.use_cosface:
            self.s = use_cosface['s']
            self.m = use_cosface['m']
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool2d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.avg_pool = None

        self.loss_reid = build_loss(loss_reid)
        self.feat = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def _refine_logits(self, x1, labels=None, dim=1, eps=1e-8):
        # 在recognizer中cls_score是无缝衔接
        x2 = self.fc_cls.weight     # c,d
        ip = torch.mm(x1, x2.t())
        w1 = torch.norm(x1, 2, dim)
        w2 = torch.norm(x2, 2, dim)
        r_logits = ip / torch.ger(w1,w2).clamp(min=eps)
        if labels is None:
            one_hot = 0
        else:
            one_hot = torch.zeros_like(r_logits)
            one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        output = self.s * (r_logits - one_hot * self.m)
        return output

    def forward(self, x, num_segs):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.
            num_segs (int): Useless in TSMHead. By default, `num_segs`
                is equal to `clip_len * num_clips * num_crops`, which is
                automatically generated in Recognizer forward phase and
                useless in TSM models. The `self.num_segments` we need is a
                hyper parameter to build TSM models.
        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N * num_segs, in_channels, 7, 7]
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        # [N * num_segs, in_channels, 1, 1]
        x = torch.flatten(x, 1)
        if self.is_shift and self.temporal_pool:
            # [2 * N, num_segs // 2, num_classes]
            x = x.view((-1, self.num_segments // 2) + x.size()[1:])
        else:
            # [N, num_segs, num_classes]
            x = x.view((-1, self.num_segments) + x.size()[1:])
        x = self.consensus(x).squeeze(1)

        # [N, in_channels]
        if self.dropout is not None:
            x = self.dropout(x)

        self.feat = x       # Following do not contain in-place operator
        if self.use_bnneck:
            x = self.bnneck(x)
        # [N, num_classes]
        if self.use_cosface:
            cls_score = x   # Just for simplity
            if not self.training:
                cls_score = self._refine_logits(cls_score)
        else:
            cls_score = self.fc_cls(x)

        # [N, num_classes]
        return cls_score

    def loss(self, cls_score, labels, **kwargs):
        """Calculate the loss given output ``cls_score``, target ``labels``.

        Args:
            cls_score (torch.Tensor): The output of the model.
            labels (torch.Tensor): The target output of the model.

        Returns:
            dict: A dict containing field 'loss_cls'(mandatory)
            and 'top1_acc', 'top5_acc'(optional).
        """
        if self.training and self.use_cosface:
            cls_score = self._refine_logits(cls_score, labels)
        losses = super().loss(cls_score, labels, **kwargs)
        loss_reid = self.loss_reid(self.feat, labels)

        if isinstance(loss_reid, dict):
            losses.update(loss_reid)
        else:
            losses['loss_reid'] = loss_reid

        return losses


@HEADS.register_module()
class FGTSMReidSimpleHead(BaseHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 num_segments=8,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 loss_reid=dict(type='TripletLoss'),
                 spatial_type='avg',
                 consensus=dict(type='AvgConsensus', dim=1),
                 dropout_ratio=0.8,
                 init_std=0.001,
                 is_shift=True,
                 temporal_pool=False,
                 use_cosface=dict(use=False, s=64, m=0.1),
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.num_segments = num_segments
        self.init_std = init_std
        self.is_shift = is_shift
        self.temporal_pool = temporal_pool

        consensus_ = consensus.copy()

        consensus_type = consensus_.pop('type')
        if consensus_type == 'AvgConsensus':
            self.consensus = AvgConsensus(**consensus_)
        else:
            self.consensus = None

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.use_cosface = use_cosface['use']
        if self.use_cosface:
            self.s = use_cosface['s']
            self.m = use_cosface['m']
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.loss_reid = build_loss(loss_reid)
        self.feat = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def _refine_logits(self, x1, labels=None, dim=1, eps=1e-8):
        # 在recognizer中cls_score是无缝衔接
        x2 = self.fc_cls.weight     # c,d
        ip = torch.mm(x1, x2.t())
        w1 = torch.norm(x1, 2, dim)
        w2 = torch.norm(x2, 2, dim)
        r_logits = ip / torch.ger(w1,w2).clamp(min=eps)
        if labels is None:
            one_hot = 0
        else:
            one_hot = torch.zeros_like(r_logits)
            one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        output = self.s * (r_logits - one_hot * self.m)
        return output

    def forward(self, x, num_segs):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.
            num_segs (int): Useless in TSMHead. By default, `num_segs`
                is equal to `clip_len * num_clips * num_crops`, which is
                automatically generated in Recognizer forward phase and
                useless in TSM models. The `self.num_segments` we need is a
                hyper parameter to build TSM models.
        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N * num_segs, in_channels, 7, 7]
        x_avg = self.avg_pool(x).flatten(1)
        x_mx = self.max_pool(x).flatten(1)

        if self.is_shift and self.temporal_pool:
            # [2 * N, num_segs // 2, num_classes]
            x_avg = x_avg.view((-1, self.num_segments // 2) + x_avg.size()[1:])
            x_mx = x_mx.view((-1, self.num_segments // 2) + x_mx.size()[1:])
        else:
            # [N, num_segs, num_classes]
            x_avg = x_avg.view((-1, self.num_segments) + x_avg.size()[1:])
            x_mx = x_mx.view((-1, self.num_segments) + x_mx.size()[1:])
        x_avg = self.consensus(x_avg).squeeze(1)
        x_mx = self.consensus(x_mx).squeeze(1)

        # [N, in_channels]
        if self.dropout is not None:
            x_avg = self.dropout(x_avg)
            x_mx = self.dropout(x_mx)

        self.feat = x_mx       # Following do not contain in-place operator
        x = x_avg
        # [N, num_classes]
        if self.use_cosface:
            cls_score = x   # Just for simplity
            if not self.training:
                cls_score = self._refine_logits(cls_score)
        else:
            cls_score = self.fc_cls(x)

        # [N, num_classes]
        return cls_score

    def loss(self, cls_score, labels, **kwargs):
        """Calculate the loss given output ``cls_score``, target ``labels``.

        Args:
            cls_score (torch.Tensor): The output of the model.
            labels (torch.Tensor): The target output of the model.

        Returns:
            dict: A dict containing field 'loss_cls'(mandatory)
            and 'top1_acc', 'top5_acc'(optional).
        """
        if self.training and self.use_cosface:
            cls_score = self._refine_logits(cls_score, labels)
        losses = super().loss(cls_score, labels, **kwargs)
        loss_reid = self.loss_reid(self.feat, labels)

        if isinstance(loss_reid, dict):
            losses.update(loss_reid)
        else:
            losses['loss_reid'] = loss_reid

        return losses