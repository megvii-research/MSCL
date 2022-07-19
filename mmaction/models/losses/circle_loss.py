# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from torch import nn
from ..builder import LOSSES
from .base import BaseWeightedLoss


"""
Multiple positive: input -> sp, sn
"""
@LOSSES.register_module()
class MultiPositiveSumLoss(BaseWeightedLoss):
    def __init__(self, loss_weight=1.0, avg_on_group=True):
        super().__init__(loss_weight=loss_weight)
        self.avg_on_group = avg_on_group

    def _forward(self, sp, sn, **kwargs):
        """Forward function.
        Args:
            sp (torch.Tensor): Similarity of positive.
            sn (torch.Tensor): Similarity of negative.
            kwargs: Any keyword argument to be used to calculate
                CrossEntropy loss.

        Returns:
            torch.Tensor: The returned CrossEntropy loss.
        """
        # sp: b, np or b, g, np
        # [lse(sn)-lse(sp)]+
        num_group = 1
        if sp.ndim == 3:
            num_group = sp.shape[1]
            sp, sn = sp.flatten(0, 1), sn.flatten(0, 1)
        assert sp.ndim == 2, f"{sp.shape}"
        loss = F.softplus(torch.logsumexp(sn, dim=1) - torch.logsumexp(sp, dim=1)).mean()
        if not self.avg_on_group:
            loss *= num_group
        return loss


@LOSSES.register_module()
class MultiPositiveUniLoss(BaseWeightedLoss):
    def __init__(self, m=0, gamma=1, loss_weight=1.0, avg_on_group=True) -> None:
        super().__init__(loss_weight=loss_weight)
        self.m = m      # margin
        self.gamma = gamma
        self.avg_on_group = avg_on_group
        self.soft_plus = nn.Softplus()

    def _forward(self, sp, sn, **kwargs):
        """
        Args:
            sp (torch.Tensor): Similarity of positive.
            sn (torch.Tensor): Similarity of negative.
            kwargs: Any keyword argument to be used to calculate
                CrossEntropy loss.
        
        Extra:
            Op, On = 1+m, -m | delta_p, delta_n = 1-m, m
            ap, an = [Op-sp]+, [sn-On]+
        """
        num_group = 1
        if sp.ndim == 3:
            num_group = sp.shape[1]
            sp, sn = sp.flatten(0, 1), sn.flatten(0, 1)

        logit_p = - sp * self.gamma
        logit_n = (sn + self.m) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=1) + torch.logsumexp(logit_p, dim=1)).mean()
        if not self.avg_on_group:
            loss *= num_group

        return loss


@LOSSES.register_module()
class MultiPositiveCircleLoss(BaseWeightedLoss):
    def __init__(self, m=0.25, gamma=128, loss_weight=1.0, avg_on_group=True) -> None:
        super().__init__(loss_weight=loss_weight)
        self.m = m      # margin
        self.gamma = gamma
        self.avg_on_group = avg_on_group
        self.soft_plus = nn.Softplus()

    def _forward(self, sp, sn, **kwargs):
        """
        Args:
            sp (torch.Tensor): Similarity of positive.
            sn (torch.Tensor): Similarity of negative.
            kwargs: Any keyword argument to be used to calculate
                CrossEntropy loss.
        
        Extra:
            Op, On = 1+m, -m | delta_p, delta_n = 1-m, m
            ap, an = [Op-sp]+, [sn-On]+
        """
        num_group = 1
        if sp.ndim == 3:
            num_group = sp.shape[1]
            sp, sn = sp.flatten(0, 1), sn.flatten(0, 1)

        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        # logit_p = - gamma * alpha_p * (s_p - delta_p) + (-99999999.) * (1 - is_pos)
        # logit_n = gamma * alpha_n * (s_n - delta_n) + (-99999999.) * (1 - is_neg)

        # loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=1) + torch.logsumexp(logit_p, dim=1)).mean()
        if not self.avg_on_group:
            loss *= num_group

        return loss