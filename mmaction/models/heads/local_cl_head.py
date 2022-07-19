import torch
from torch import nn as nn


from ..builder import HEADS, build_loss
from .base import BaseHead
from ...core import top_k_accuracy


@HEADS.register_module()
class MSCLWithAugPosHeadV2(BaseHead):
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