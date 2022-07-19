from collections import defaultdict
import torch
from torch import nn as nn
import torch.nn.functional as F

from torch.nn.modules import loss

from mmaction.models.recognizers.mscl import forward

from ..builder import HEADS, build_loss, build_head
from .base import BaseHead
from ...core import top_k_accuracy, bbox_overlaps
from .base import create_adaptive_pooling_3d

@HEADS.register_module()
class MSCLWithAugMxHead(BaseHead):
    """ Head for MoCo.

    Args:
        basename (str): For distinguish losses in log.
    """

    def __init__(
        self, basename='', loss_cls=dict(type="CrossEntropyLoss"),
        num_classes=2, in_channels=128, same_kn=True, T=0.07,
    ):
        super().__init__(loss_cls=loss_cls, num_classes=num_classes, in_channels=in_channels)
        if basename:
            basename = '_' + basename
        self.basename = basename
        self.same_kn = same_kn
        self.T = T

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

        ssl_label = torch.zeros(rf_logits.shape[0], dtype=torch.long).cuda()
        return rf_logits, fr_logits, ssl_label

    def _loss_mx(self, cls_score, labels, basename=None, **kwargs):
        """Calculate the loss given output ``cls_score``, target ``labels``.

        Args:
            cls_score (torch.Tensor): The output of the model.
            labels (torch.Tensor): The target output of the model.

        Returns:
            dict: A dict containing field 'loss_cls'(mandatory)
            and 'top1_acc', 'top5_acc'(optional).
        """
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
class MSCLWithAugSimpleHead(BaseHead):
    def __init__(self, loss_cls=dict(type="CrossEntropyLoss"),
                 num_classes=2, in_channels=128,):
        super().__init__(loss_cls=loss_cls, num_classes=num_classes, in_channels=in_channels)

    def init_weights(self):
        pass

    def loss(self, **kwargs):
        return dict()

    def forward(self, **kwargs):
        return dict()

    def update_aux_info(self, info_name, info_dict, target):
        target


@HEADS.register_module()
class MoDistv2PosHead(BaseHead):
    def __init__(self, basename='', loss_cls=dict(type="CrossEntropyLoss"),
                 loss_pos=dict(type="CrossEntropyLoss"),
                 num_classes=2, in_channels=128, mlvl_ids=(0, -1),
                 bkb_channels=(512, 128), t=8, T=0.07, aux_keys=dict()):
        super().__init__(loss_cls=loss_cls, num_classes=num_classes, in_channels=in_channels)
        self.loss_pos = build_loss(loss_pos)
        if basename:
            basename = '_' + basename
        self.basename = basename
        self.rgb_pooling = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.flow_pooling = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.T = T
        self.aux_keys = aux_keys
        self.mlvl_ids = mlvl_ids

        if bkb_channels[0] is not None:
            self.trans_rgb = nn.Sequential(nn.Conv1d(bkb_channels[0], 128, 1), nn.ReLU(), nn.Conv1d(128, 128, 1))
        else:
            self.trans_rgb = nn.Identity()
        self.trans_flow = nn.Conv1d(bkb_channels[1], 128, 1)
        labels = torch.arange(t).unsqueeze(0)
        self.register_buffer('labels', labels)

    def init_weights(self):
        pass

    def _loss_pos(self, pos_scores, pos_labels, **kwargs):
        # loss for postion prediction
        losses = dict()
        losses['loss_pos'] = self.loss_pos(pos_scores, pos_labels)
        top_k_acc = top_k_accuracy(pos_scores.detach().cpu().numpy(),
                                    pos_labels.detach().cpu().numpy(), (1, 5))
        losses['top1_acc_pos'] = torch.tensor(
            top_k_acc[0], device=pos_scores.device)
        losses['top5_acc_pos'] = torch.tensor(
            top_k_acc[1], device=pos_scores.device)
        return losses

    def loss(self, pos_scores, pos_labels, **kwargs):
        losses_pos = self._loss_pos(pos_scores, pos_labels)
        return losses_pos

    def forward(self, q_mlvl, q_flow_mlvl, **kwargs):
        x_q = q_mlvl[self.mlvl_ids[0]]
        x_q_flow = q_flow_mlvl[self.mlvl_ids[1]]

        x_q = self.rgb_pooling(x_q).view(*x_q.shape[:-2])   # b,c,t
        x_q_flow = self.flow_pooling(x_q_flow).view(*x_q_flow.shape[:-2])   # b,c,2t

        if hasattr(self, 'trans_rgb'):
            x_q = self.trans_rgb(x_q)
        x_q_flow = self.trans_flow(x_q_flow)
        x_q = nn.functional.normalize(x_q, dim=1)
        x_q_flow = nn.functional.normalize(x_q_flow, dim=1)
        sim = torch.bmm(x_q.transpose(1, 2), x_q_flow)  # b,t,t

        pos_scores = sim.flatten(0, 1)/self.T
        pos_labels = self.labels.repeat((x_q.shape[0], 1)).flatten(0, 1)
        return dict(pos_scores=pos_scores, pos_labels=pos_labels)

    def update_aux_info(self, info_name, info_dict, target):
        if info_name in self.aux_keys:
            for k in self.aux_keys[info_name]:
                assert self.aux_keys[info_name][k] not in target, \
                    f"Find key-{self.aux_keys[info_name][k]} in target dict with keys:{target.keys()}"
                target[self.aux_keys[info_name][k]] = info_dict[k]
        return target

@HEADS.register_module()
class MSCLWithAugPosHead(BaseHead):
    def __init__(self, basename='', loss_cls=dict(type="CrossEntropyLoss"),
                 loss_pos=dict(type="CrossEntropyLoss"),
                 num_classes=2, in_channels=128, mlvl_ids=(0, -1),
                 bkb_channels=(512, 128), t=8, T=0.07, aux_keys=dict()):
        super().__init__(loss_cls=loss_cls, num_classes=num_classes, in_channels=in_channels)
        self.loss_pos = build_loss(loss_pos)
        if basename:
            basename = '_' + basename
        self.basename = basename
        self.rgb_pooling = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.flow_pooling = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.T = T
        self.aux_keys = aux_keys
        self.mlvl_ids = mlvl_ids

        if bkb_channels[0] is not None:
            self.trans_rgb = nn.Sequential(nn.Conv1d(bkb_channels[0], 128, 1), nn.ReLU(), nn.Conv1d(128, 128, 1))
        else:
            self.trans_rgb = nn.Identity()
        self.trans_flow = nn.Conv1d(bkb_channels[1], 128, 1)
        labels = torch.arange(t).unsqueeze(0)
        self.register_buffer('labels', labels)

    def init_weights(self):
        pass

    def _loss_pos(self, pos_scores, pos_labels, **kwargs):
        # loss for postion prediction
        losses = dict()
        losses['loss_pos'] = self.loss_pos(pos_scores, pos_labels)
        top_k_acc = top_k_accuracy(pos_scores.detach().cpu().numpy(),
                                    pos_labels.detach().cpu().numpy(), (1, 5))
        losses['top1_acc_pos'] = torch.tensor(
            top_k_acc[0], device=pos_scores.device)
        losses['top5_acc_pos'] = torch.tensor(
            top_k_acc[1], device=pos_scores.device)
        return losses

    def loss(self, pos_scores, pos_labels, **kwargs):
        losses_pos = self._loss_pos(pos_scores, pos_labels)
        return losses_pos

    def forward(self, q_mlvl, q_flow_mlvl, q_aug_flow_mlvl, **kwargs):
        x_q = q_mlvl[self.mlvl_ids[0]]
        x_q_flow = torch.cat((q_flow_mlvl[self.mlvl_ids[1]], q_aug_flow_mlvl[self.mlvl_ids[1]]), dim=2)

        x_q = self.rgb_pooling(x_q).view(*x_q.shape[:-2])   # b,c,t
        x_q_flow = self.flow_pooling(x_q_flow).view(*x_q_flow.shape[:-2])   # b,c,2t

        if hasattr(self, 'trans_rgb'):
            x_q = self.trans_rgb(x_q)
        x_q_flow = self.trans_flow(x_q_flow)
        x_q = nn.functional.normalize(x_q, dim=1)
        x_q_flow = nn.functional.normalize(x_q_flow, dim=1)
        sim = torch.bmm(x_q.transpose(1, 2), x_q_flow)  # b,t,t

        pos_scores = sim.flatten(0, 1)/self.T
        pos_labels = self.labels.repeat((x_q.shape[0], 1)).flatten(0, 1)
        return dict(pos_scores=pos_scores, pos_labels=pos_labels)

    def update_aux_info(self, info_name, info_dict, target):
        if info_name in self.aux_keys:
            for k in self.aux_keys[info_name]:
                assert self.aux_keys[info_name][k] not in target, \
                    f"Find key-{self.aux_keys[info_name][k]} in target dict with keys:{target.keys()}"
                target[self.aux_keys[info_name][k]] = info_dict[k]
        return target


@HEADS.register_module()
class MSCLWithAugAPPosHead(BaseHead):
    def __init__(self, basename='', loss_cls=dict(type="CrossEntropyLoss"),
                 loss_pos=dict(type="CrossEntropyLoss"),
                 num_classes=2, in_channels=128, mlvl_ids=(0, -1), num_ap=8,
                 bkb_channels=(512, 128), t=8, T=0.07, aux_keys=dict()):
        super().__init__(loss_cls=loss_cls, num_classes=num_classes, in_channels=in_channels)
        self.loss_pos = build_loss(loss_pos)
        if basename:
            basename = '_' + basename
        self.basename = basename
        self.rgb_pooling = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.flow_pooling = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.T = T
        self.aux_keys = aux_keys
        self.mlvl_ids = mlvl_ids

        if bkb_channels[0] is not None:
            raise NotImplementedError("Not support rgb feature without FPN!!")
            # self.trans_rgb = nn.Sequential(nn.Conv1d(bkb_channels[0], 128, 1), nn.ReLU(), nn.Conv1d(128, 128, 1))
        else:
            self.trans_rgb = nn.Identity()
            self.angel_prediction = nn.Sequential(nn.Linear(bkb_channels[1]*2, 128), nn.ReLU(), nn.Linear(128, num_ap))
        self.trans_flow = nn.Conv1d(bkb_channels[1], 128, 1)
        labels = torch.arange(t).unsqueeze(0)
        self.register_buffer('labels', labels)

    def init_weights(self):
        pass

    def _loss_pos(self, pos_scores, pos_labels, **kwargs):
        # loss for postion prediction
        losses = dict()
        losses['loss_pos'] = self.loss_pos(pos_scores, pos_labels)
        top_k_acc = top_k_accuracy(pos_scores.detach().cpu().numpy(),
                                    pos_labels.detach().cpu().numpy(), (1, 5))
        losses['top1_acc_pos'] = torch.tensor(
            top_k_acc[0], device=pos_scores.device)
        losses['top5_acc_pos'] = torch.tensor(
            top_k_acc[1], device=pos_scores.device)
        return losses

    def loss(self, pos_scores, pos_labels, ap_scores, ap_labels, **kwargs):
        losses = self._loss_pos(pos_scores, pos_labels)
        loss_ap = self.loss_cls(ap_scores, ap_labels)
        if isinstance(loss_ap, dict):
            losses.update(loss_ap)
        else:
            losses['loss_ap'] = loss_ap
        return losses

    def forward(self, q_mlvl, q_flow_mlvl, q_aug_flow_mlvl, **kwargs):
        q_ap, q_mlvl = q_mlvl[-1], q_mlvl[:-1]
        _, q_flow_mlvl = q_flow_mlvl[-1], q_flow_mlvl[:-1]
        q_aug_flow_ap, q_aug_flow_mlvl = q_aug_flow_mlvl[-1], q_aug_flow_mlvl[:-1]
        ap_scores = self.angel_prediction(torch.cat([q_ap, q_aug_flow_ap], dim=-1))

        x_q = q_mlvl[self.mlvl_ids[0]]
        x_q_flow = torch.cat((q_flow_mlvl[self.mlvl_ids[1]], q_aug_flow_mlvl[self.mlvl_ids[1]]), dim=2)

        x_q = self.rgb_pooling(x_q).view(*x_q.shape[:-2])   # b,c,t
        x_q_flow = self.flow_pooling(x_q_flow).view(*x_q_flow.shape[:-2])   # b,c,2t

        if hasattr(self, 'trans_rgb'):
            x_q = self.trans_rgb(x_q)
        x_q_flow = self.trans_flow(x_q_flow)
        x_q = nn.functional.normalize(x_q, dim=1)
        x_q_flow = nn.functional.normalize(x_q_flow, dim=1)
        sim = torch.bmm(x_q.transpose(1, 2), x_q_flow)  # b,t,t

        pos_scores = sim.flatten(0, 1)/self.T
        pos_labels = self.labels.repeat((x_q.shape[0], 1)).flatten(0, 1)
        return dict(pos_scores=pos_scores, pos_labels=pos_labels, ap_scores=ap_scores)

    def update_aux_info(self, info_name, info_dict, target):
        if info_name in self.aux_keys:
            for k in self.aux_keys[info_name]:
                assert self.aux_keys[info_name][k] not in target, \
                    f"Find key-{self.aux_keys[info_name][k]} in target dict with keys:{target.keys()}"
                target[self.aux_keys[info_name][k]] = info_dict[k]
        return target


@HEADS.register_module()
class MlvlMSCLWithAugPosHead(BaseHead):
    def __init__(self, basename='', loss_cls=dict(type="CrossEntropyLoss"),
                 loss_pos=dict(type="CrossEntropyLoss"),
                 num_classes=2, in_channels=128, mlvl_ids=(0, 1, 2), mlvl_flow_ids=(-1, -1, -1),
                 bkb_channels=(None, None), t=8, T=0.07, pool_type='avg', aux_keys=dict()):
        super().__init__(loss_cls=loss_cls, num_classes=num_classes, in_channels=in_channels)
        self.loss_pos = build_loss(loss_pos)
        if basename:
            basename = '_' + basename
        self.basename = basename
        self.rgb_pooling = create_adaptive_pooling_3d(pool_type, (None, 1, 1))
        self.flow_pooling = create_adaptive_pooling_3d(pool_type, (None, 1, 1))
        self.T = T
        self.aux_keys = aux_keys
        self.mlvl_ids = mlvl_ids
        self.mlvl_flow_ids = mlvl_flow_ids
        self.num_ids = len(mlvl_ids)

        # 这里是共享的projection，非共享的在Neck中定义
        if bkb_channels[0] is not None:
            self.trans_rgb = nn.Conv1d(bkb_channels[0], 128, 1)
        else:
            self.trans_rgb = nn.Identity()
        if bkb_channels[1] is not None:
            self.trans_flow = nn.Conv1d(bkb_channels[1], 128, 1)
        else:
            self.trans_flow = nn.Identity()
        labels = torch.arange(t).unsqueeze(0)
        self.register_buffer('labels', labels)

    def init_weights(self):
        pass

    def _loss_pos(self, pos_scores, pos_labels, **kwargs):
        # loss for postion prediction
        losses = defaultdict(lambda: 0)
        losses['loss_pos'] = self.loss_pos(pos_scores, pos_labels)/self.num_ids
        top_k_acc = top_k_accuracy(pos_scores.detach().cpu().numpy(),
                                    pos_labels.detach().cpu().numpy(), (1, 5))
        losses['top1_acc_pos'] = torch.tensor(
            top_k_acc[0], device=pos_scores.device)
        losses['top5_acc_pos'] = torch.tensor(
            top_k_acc[1], device=pos_scores.device)
        return losses

    def loss(self, pos_scores, pos_labels, **kwargs):
        losses_pos = dict()
        for i, (pos_score, pos_label) in enumerate(zip(pos_scores, pos_labels)):
            loss_pos = self._loss_pos(pos_score, pos_label)
            for k, v in loss_pos.items():
                losses_pos[k + f'_{i}'] = v
        return losses_pos
        
    def forward(self, q_mlvl, q_flow_mlvl, q_aug_flow_mlvl=None, **kwargs):
        pos_scores, pos_labels = list(), list()
        for rgb_id, flow_id in zip(self.mlvl_ids, self.mlvl_flow_ids):
            x_q = q_mlvl[rgb_id]
            if q_aug_flow_mlvl is not None:
                x_q_flow = torch.cat((q_flow_mlvl[flow_id], q_aug_flow_mlvl[flow_id]), dim=2)
            else:
                x_q_flow = q_flow_mlvl[flow_id]
            pos_score, pos_label = self._forward_single(x_q, x_q_flow)
            pos_scores.append(pos_score)
            pos_labels.append(pos_label)

        return dict(pos_scores=pos_scores, pos_labels=pos_labels)


    def _forward_single(self, x_q, x_q_flow, **kwargs):
        x_q = self.rgb_pooling(x_q).view(*x_q.shape[:-2])   # b,c,t
        x_q_flow = self.flow_pooling(x_q_flow).view(*x_q_flow.shape[:-2])   # b,c,2t

        x_q = self.trans_rgb(x_q)
        x_q_flow = self.trans_flow(x_q_flow)
        x_q = nn.functional.normalize(x_q, dim=1)
        x_q_flow = nn.functional.normalize(x_q_flow, dim=1)
        sim = torch.bmm(x_q.transpose(1, 2), x_q_flow)  # b,t,t

        pos_score = sim.flatten(0, 1)/self.T
        pos_label = self.labels.repeat((x_q.shape[0], 1)).flatten(0, 1)
        return pos_score, pos_label

    def update_aux_info(self, info_name, info_dict, target):
        if info_name in self.aux_keys:
            for k in self.aux_keys[info_name]:
                assert self.aux_keys[info_name][k] not in target, \
                    f"Find key-{self.aux_keys[info_name][k]} in target dict with keys:{target.keys()}"
                target[self.aux_keys[info_name][k]] = info_dict[k]
        return target


@HEADS.register_module()
class MAMSCLWithAugPosHead(BaseHead):
    def __init__(self, basename='', loss_cls=dict(type="CrossEntropyLoss"),
                 loss_pos=dict(type="CrossEntropyLoss"),
                 num_classes=2, in_channels=128,
                 bkb_channels=(512, 128), t=8, T=0.07, aux_keys=dict(),
                 chosen_rate=0.2):
        super().__init__(loss_cls=loss_cls, num_classes=num_classes, in_channels=in_channels)
        self.loss_pos = build_loss(loss_pos)
        if basename:
            basename = '_' + basename
        self.basename = basename
        self.rgb_pooling = nn.AvgPool3d(kernel_size=(1, 4, 4), stride=(1, 4, 4))  # 28 -> 7
        self.flow_pooling = nn.Identity()
        self.T = T
        self.aux_keys = aux_keys
        self.chosen_rate = chosen_rate

        if bkb_channels[0] is not None:
            self.trans_rgb = nn.Conv1d(bkb_channels[0], 128, 1)
        else:
            self.trans_rgb = nn.Identity()
        self.trans_flow = nn.Conv3d(bkb_channels[1], 128, 1)
        labels = torch.arange(t).unsqueeze(0)
        self.register_buffer('labels', labels)

    def init_weights(self):
        pass

    def _cal_weight(self, motion_map: torch.Tensor):
        # motion_map: b,t,h,w
        assert motion_map.requires_grad is False
        b, t, h, w = motion_map.shape
        motion_map = motion_map.flatten(-2, -1)
        K = max(int(h*w*self.chosen_rate), 1)
        _, topk_inds = motion_map.topk(k=K, dim=-1)     # b,t,k
        # 当索引的维度与被索引张量的维度一致时，往往用的是scatter/gather
        weight = motion_map.new_zeros((b, t, h*w))
        weight = weight.scatter(dim=-1, index=topk_inds, value=1)
        weight = weight.unflatten(-1, (h, w))
        return weight

    def _loss_pos(self, pos_scores, pos_labels, **kwargs):
        # loss for postion prediction
        losses = dict()
        losses['loss_pos'] = self.loss_pos(pos_scores, pos_labels)
        top_k_acc = top_k_accuracy(pos_scores.detach().cpu().numpy(),
                                    pos_labels.detach().cpu().numpy(), (1, 5))
        losses['top1_acc_pos'] = torch.tensor(
            top_k_acc[0], device=pos_scores.device)
        losses['top5_acc_pos'] = torch.tensor(
            top_k_acc[1], device=pos_scores.device)
        return losses

    def loss(self, pos_scores, pos_labels, **kwargs):
        losses_pos = self._loss_pos(pos_scores, pos_labels)
        return losses_pos

    def forward(self, q_mlvl, q_flow_mlvl, motion_maps_q, **kwargs):
        x_q = q_mlvl[0]
        x_q_flow = q_flow_mlvl[-1]

        x_q = self.rgb_pooling(x_q)   # b,c,t,h,w
        x_q_flow = self.flow_pooling(x_q_flow)
        new_t, new_h, new_w = x_q.shape[-3:]
        # x_q = x_q[..., :7, :7]   # b,c,t,h,w
        # x_q_flow = x_q_flow[..., :7, :7]

        # For simplicity, trans_rgb is not used(default to 128 dim...)
        x_q_flow = self.trans_flow(x_q_flow)
        x_q = nn.functional.normalize(x_q, dim=1)
        x_q_flow = nn.functional.normalize(x_q_flow, dim=1)
        x_q = x_q.permute(0, 3, 4, 2, 1).flatten(0, 2)  # bhw,t,c
        x_q_flow = x_q_flow.permute(0, 3, 4, 1, 2).flatten(0, 2)    # bhw,c,t

        sim = torch.bmm(x_q, x_q_flow)  # bhw,t,t

        pos_scores = sim.flatten(0, 1)/self.T
        pos_labels = self.labels.repeat((x_q.shape[0], 1)).flatten(0, 1)
        motion_weight = self._cal_weight(motion_maps_q)  # b,t,h,w
        motion_weight = motion_weight.unsqueeze(1)  # b,1,t,h,w
        motion_weight = F.adaptive_avg_pool3d(motion_weight, (new_t, new_h, new_w))
        motion_weight = motion_weight.squeeze(1).permute(0, 2, 3, 1).reshape(-1)  # bhwt

        return dict(pos_scores=pos_scores, pos_labels=pos_labels, motion_weight=motion_weight)

    def update_aux_info(self, info_name, info_dict, target):
        if info_name in self.aux_keys:
            for k in self.aux_keys[info_name]:
                assert self.aux_keys[info_name][k] not in target, \
                    f"Find key-{self.aux_keys[info_name][k]} in target dict with keys:{target.keys()}"
                target[self.aux_keys[info_name][k]] = info_dict[k]
        return target
