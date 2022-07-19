from collections import defaultdict
import torch
from torch import nn as nn
import torch.nn.functional as F

from torch.nn.modules import loss

from mmaction2.mmaction.models.recognizers.mscl import forward

from ..builder import HEADS, build_loss, build_head
from .base import BaseHead
from ...core import top_k_accuracy, bbox_overlaps
from .base import create_adaptive_pooling_3d

@HEADS.register_module()
class MoCoHeadV2(BaseHead):
    """ MoCo Head for MoCoV2.
    Support forward with previous features.
    """

    def __init__(
        self, basename='', loss_cls=dict(type="CrossEntropyLoss"),
        num_classes=2, in_channels=128, T=0.07,
    ):
        super().__init__(loss_cls=loss_cls, num_classes=num_classes, in_channels=in_channels)
        if basename:
            basename = '_' + basename
        self.basename = basename
        self.T = T

    def init_weights(self):
        """Initiate the parameters from scratch."""
        pass

    def forward(self, q, k, weight, **kwargs):
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q, weight])
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # ssl_label: positive key indicators
        ssl_label = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        return dict(cls_score=logits, ssl_label=ssl_label)

    def extract_global_feat(x):
        pass

    def loss(self, cls_score, ssl_label, basename=None, **kwargs):
        if basename is None:
            basename = self.basename
        losses = dict()
        if ssl_label.shape == torch.Size([]):
            ssl_label = ssl_label.unsqueeze(0)
        elif ssl_label.dim() == 1 and ssl_label.size()[0] == self.num_classes \
                and cls_score.size()[0] == 1:
            ssl_label = ssl_label.unsqueeze(0)

        if not self.multi_class and cls_score.size() != ssl_label.size():
            top_k_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                       ssl_label.detach().cpu().numpy(), (1, 5))
            losses[f'top1_acc{basename}'] = torch.tensor(
                top_k_acc[0], device=cls_score.device)
            losses[f'top5_acc{basename}'] = torch.tensor(
                top_k_acc[1], device=cls_score.device)

        elif self.multi_class and self.label_smooth_eps != 0:
            ssl_label = ((1 - self.label_smooth_eps) * ssl_label +
                      self.label_smooth_eps / self.num_classes)

        loss_cls = self.loss_cls(cls_score, ssl_label)
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses[f'loss_cls{basename}'] = loss_cls

        return losses

    def loss_mx(self, **kwargs):
        # loss for postion prediction
        return dict()


@HEADS.register_module()
class MSFHead(BaseHead):
    """ MSF Head for MoCoV2.
    Support forward with previous features.
    """

    def __init__(
        self, basename='', loss_cls=dict(type="CrossEntropyLoss"),
        num_classes=2, in_channels=128, topk=5,
    ):
        super().__init__(loss_cls=loss_cls, num_classes=num_classes, in_channels=in_channels)
        if basename:
            basename = '_' + basename
        self.basename = basename
        self.topk = topk

    def init_weights(self):
        """Initiate the parameters from scratch."""
        pass

    def forward(self, q, k, weight, label, label_queue, **kwargs):
        """Mean Shift for SSL
        https://github.com/UMBCvision/MSF
        """
        # weight is .clone().detach() before, shape is (c, k)
        dist_t = 2 - 2 * torch.einsum('bc,ck->bk', [k, weight]) # current_target, targets
        dist_q = 2 - 2 * torch.einsum('bc,ck->bk', [q, weight]) # query, targets

        # select the k nearest neighbors [with smallest distance (largest=False)] based on current target
        _, nn_index = dist_t.topk(self.topk, dim=1, largest=False)
        nn_dist_q = torch.gather(dist_q, 1, nn_index)   # b, topk

        label = label.unsqueeze(1).expand(nn_dist_q.shape[0], nn_dist_q.shape[1])
        mem_bank_size = label_queue.shape[0]
        label_queue = label_queue.unsqueeze(0).expand((nn_dist_q.shape[0], mem_bank_size))
        label_queue = torch.gather(label_queue, dim=1, index=nn_index)
        matches = (label_queue == label).float()

        return dict(nn_dist_q=nn_dist_q, matches=matches)

    def extract_global_feat(x):
        pass

    def loss(self, nn_dist_q, matches, basename=None, **kwargs):
        if basename is None:
            basename = self.basename
        losses = dict()

        loss_msf = (nn_dist_q.sum(dim=1) / self.topk).mean()
        purity = (matches.sum(dim=1) / self.topk).mean()
        losses[f'msf_purity{basename}'] = purity
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_msf, dict):
            losses.update(loss_msf)
        else:
            losses[f'loss_msf{basename}'] = loss_msf

        return losses

    def loss_mx(self, **kwargs):
        # loss for postion prediction
        return dict()


@HEADS.register_module()
class NMSFHead(BaseHead):
    """ NMSF Head for MoCoV2. NMSF -> MSF with negative
    Support forward with previous features.
    """

    def __init__(
        self, basename='', loss_cls=dict(type="MultiPositiveSumLoss"),
        num_classes=2, in_channels=128, T=0.07, topk=5, pos_type='sum',
    ):
        super().__init__(loss_cls=loss_cls, num_classes=num_classes, in_channels=in_channels)
        if basename:
            basename = '_' + basename
        self.basename = basename
        self.T = T
        self.topk = topk
        self.pos_type = pos_type

    def init_weights(self):
        """Initiate the parameters from scratch."""
        pass

    def forward(self, q, k, weight, label, label_queue, **kwargs):
        """Mean Shift for SSL
        https://github.com/UMBCvision/MSF
        """
        # weight is .clone().detach() before, shape is (c, k)
        dist_t = torch.einsum('bc,ck->bk', [k, weight]) # current_target, targets
        dist_q = torch.einsum('bc,ck->bk', [q, weight]) # query, targets

        # select the k nearest neighbors [with smallest distance (largest=False)] based on current target
        _, nn_index = dist_t.topk(self.topk, dim=1, largest=True)
        dist_q /= self.T
        sp = torch.gather(dist_q, 1, nn_index)   # b, topk
        mask = torch.zeros_like(dist_q).scatter(dim=1, index=nn_index, value=1.0)   # b,k
        sn = dist_q*(1-mask) + (-1e6)*mask

        label = label.unsqueeze(1).expand(dist_q.shape[0], self.topk)
        mem_bank_size = label_queue.shape[0]
        label_queue = label_queue.unsqueeze(0).expand((dist_q.shape[0], mem_bank_size))
        label_queue = torch.gather(label_queue, dim=1, index=nn_index)
        matches = (label_queue == label).float()

        return dict(sp=sp, sn=sn, matches=matches)

    def extract_global_feat(x):
        pass

    def loss(self, sp, sn, matches, basename=None, **kwargs):
        if basename is None:
            basename = self.basename
        losses = dict()
        loss_cls = self.loss_cls(sp, sn)
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses[f'loss_circle{basename}'] = loss_cls

        purity = (matches.sum(dim=1) / self.topk).mean()
        losses[f'msf_purity{basename}'] = purity

        return losses

    def loss_mx(self, **kwargs):
        # loss for postion prediction
        return dict()


@HEADS.register_module()
class MSCLWithAugMSFMxHead(BaseHead):
    """ Head for MSCLWithAug.
    Simple Implementation of MSF for distill
    """

    def __init__(
        self, basename='', loss_cls=dict(type="MultiPositiveSumLoss"),
        num_classes=2, in_channels=128, same_kn=True, T=0.07, topk=5, pos_type='sum',
    ):
        super().__init__(loss_cls=loss_cls, num_classes=num_classes, in_channels=in_channels)
        if basename:
            basename = '_' + basename
        self.basename = basename
        self.same_kn = same_kn
        self.T = T
        self.topk = topk
        self.pos_type = pos_type

    def init_weights(self):
        """Initiate the parameters from scratch."""
        pass

    def _cal_msf(self, q, k, weight, **kwargs):
        # weight is .clone().detach() before, shape is (c, k)
        dist_t = torch.einsum('bc,ck->bk', [k, weight]) # current_target, targets
        dist_q = torch.einsum('bc,ck->bk', [q, weight]) # query, targets

        # select the k nearest neighbors [with smallest distance (largest=False)] based on current target
        _, nn_index = dist_t.topk(self.topk, dim=1, largest=True)
        dist_q /= self.T
        sp = torch.gather(dist_q, 1, nn_index)   # b, topk
        mask = torch.zeros_like(dist_q).scatter(dim=1, index=nn_index, value=1.0)   # b,k
        sn = dist_q*(1-mask) + (-1e6)*mask

        return sp, sn

    def _forward_moco_mx(self, q, k, q_flow, k_flow, weight, weight_flow, **kwargs):
        # q, q_mlvl, k, k_mlvl
        if self.same_kn:
            rf_logits = self._cal_msf(q, k_flow, weight_flow)
            fr_logits = self._cal_msf(q_flow, k, weight)
        else:
            rf_logits = self._cal_msf(q, k_flow, weight)
            fr_logits = self._cal_msf(q_flow, k, weight_flow)

        ssl_label = torch.zeros(rf_logits[0].shape[0], dtype=torch.long).cuda()
        return rf_logits, fr_logits, ssl_label

    def _loss_mx(self, logits, labels, basename=None, **kwargs):
        if basename is None:
            basename = self.basename
        losses = dict()

        loss_cls = self.loss_cls(*logits)
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses[f'loss_circle{basename}'] = loss_cls

        return losses

    def loss(self, rf_logits, fr_logits, ssl_label, suffix=''):
        losses = self._loss_mx(rf_logits, ssl_label, basename=self.basename+suffix)
        losses_r = self._loss_mx(fr_logits, ssl_label, basename=self.basename+'_r'+suffix)
        losses.update(losses_r)
        return losses

    def forward(self, **kwargs):
        pass

    def extract_global_feat(x):
        pass


@HEADS.register_module()
class MSCLWithAugDistillMxHead(BaseHead):
    """ Head for MSCLWithAug.
    Simple Implementation of distribution distill.
    """
    def __init__(
        self, basename='', loss_cls=dict(type="CrossEntropyLoss"),
        num_classes=2, in_channels=128, same_kn=True, T=0.07, small_p=None,
    ):
        super().__init__(loss_cls=loss_cls, num_classes=num_classes, in_channels=in_channels)
        if basename:
            basename = '_' + basename
        self.basename = basename
        self.same_kn = same_kn
        self.T = T
        self.small_p = small_p
        self.kld = nn.KLDivLoss(reduction='batchmean')

    def init_weights(self):
        """Initiate the parameters from scratch."""
        pass

    def _forward_moco_mx(self, q, k, q_flow, k_flow, weight, weight_flow):
        # q, q_mlvl, k, k_mlvl
        rf_l_pos = torch.einsum("nc,nc->n", [q, k_flow]).unsqueeze(-1)
        fr_l_pos = torch.einsum("nc,nc->n", [q_flow, k]).unsqueeze(-1)
        if self.same_kn:
            rf_l_neg = torch.einsum("nc,ck->nk", [q, weight_flow])
            fr_l_neg = torch.einsum("nc,ck->nk", [q_flow, weight])
        else:
            rf_l_neg = torch.einsum("nc,ck->nk", [q, weight])
            fr_l_neg = torch.einsum("nc,ck->nk", [q_flow, weight_flow])

        rf_logits = torch.cat([rf_l_pos, rf_l_neg], dim=1)/self.T
        fr_logits = torch.cat([fr_l_pos, fr_l_neg], dim=1)/self.T

        p_rgb = torch.einsum("nc,ck->nk", [q, weight])
        p_flow = torch.einsum("nc,ck->nk", [q_flow, weight_flow])
        if self.small_p is not None:
            dist_t = 2 - 2 * p_flow

            _, nn_index = dist_t.topk(self.small_p, dim=1, largest=False)
            p_rgb = torch.gather(p_rgb, 1, nn_index) 
            p_flow = torch.gather(p_flow, 1, nn_index)
        p_rgb = p_rgb.softmax(dim=-1)
        p_flow = p_flow.softmax(dim=-1)
        
        ssl_label = torch.zeros(rf_logits.shape[0], dtype=torch.long).cuda()
        loss_kl = self.kld(p_rgb.log(), p_flow)
        return rf_logits, fr_logits, ssl_label, dict(loss_kl=loss_kl)

    def _loss_mx(self, cls_score, labels, basename=None, **kwargs):
        if basename is None:
            basename = self.basename
        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
                and cls_score.size()[0] == 1:
            labels = labels.unsqueeze(0)

        if not self.multi_class and cls_score.size() != labels.size():
            top_k_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                       labels.detach().cpu().numpy(), (1, 5))
            losses[f'top1_acc{basename}'] = torch.tensor(
                top_k_acc[0], device=cls_score.device)
            losses[f'top5_acc{basename}'] = torch.tensor(
                top_k_acc[1], device=cls_score.device)

        elif self.multi_class and self.label_smooth_eps != 0:
            labels = ((1 - self.label_smooth_eps) * labels +
                      self.label_smooth_eps / self.num_classes)

        loss_cls = self.loss_cls(cls_score, labels)
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses[f'loss_cls{basename}'] = loss_cls

        return losses

    def loss(self, rf_logits, fr_logits, ssl_label, loss_kl, suffix='', **kwargs):
        losses = self._loss_mx(rf_logits, ssl_label, basename=self.basename+suffix)
        losses_r = self._loss_mx(fr_logits, ssl_label, basename=self.basename+'_r'+suffix)
        losses.update(losses_r)
        losses['loss_kl'] = loss_kl
        return losses

    def forward(self, **kwargs):
        pass

    def extract_global_feat(x):
        pass