import torch
from torch import nn as nn

from ..builder import HEADS, build_loss, build_head
from .base import BaseHead
from ...core import top_k_accuracy, bbox_overlaps


@HEADS.register_module()
class MoCoHead(BaseHead):
    """ Head for MoCo.

    Args:
        basename (str): For distinguish losses in log.
    """

    def __init__(
        self, basename='', loss_cls=dict(type="CrossEntropyLoss"),
        num_classes=2, in_channels=128,
    ):
        super().__init__(loss_cls=loss_cls, num_classes=num_classes, in_channels=in_channels)
        if basename:
            basename = '_' + basename
        self.basename = basename

    def init_weights(self):
        """Initiate the parameters from scratch."""
        pass

    def forward(self, **kwargs):
        """nothing qaq
        """
        return dict()

    def extract_global_feat(x):
        pass

    def loss(self, cls_score, labels, basename=None, **kwargs):
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

    def loss_mx(self, **kwargs):
        # loss for postion prediction
        return dict()


@HEADS.register_module()
class MoDistPredHead(BaseHead):
    """ Head for MoDistPred
    """

    def __init__(
        self, basename='', loss_cls=dict(type="CrossEntropyLoss"),
        loss_pos=dict(type="CrossEntropyLoss"),
        num_classes=2, in_channels=128, bkb_channels=(512, 128), t=8, T=0.07,
    ):
        super().__init__(loss_cls=loss_cls, num_classes=num_classes, in_channels=in_channels)
        if basename:
            basename = '_' + basename
        self.loss_pos = build_loss(loss_pos)
        self.basename = basename
        self.rgb_pooling = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.flow_pooling = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.T = T

        if bkb_channels[0] is not None:
            self.trans_rgb = nn.Conv1d(bkb_channels[0], 128, 1)
        else:
            self.trans_rgb = nn.Identity()
        self.trans_flow = nn.Conv1d(bkb_channels[1], 128, 1)
        labels = torch.arange(t).unsqueeze(0)
        self.register_buffer('labels', labels)


    def init_weights(self):
        """Initiate the parameters from scratch."""
        pass

    def forward(self, q_mlvl, q_flow_mlvl, **kwargs):
        x_q = q_mlvl[0]
        x_q_flow = q_flow_mlvl[-1]

        x_q = self.rgb_pooling(x_q).view(*x_q.shape[:-2])   # b,c,t
        x_q_flow = self.flow_pooling(x_q_flow).view(*x_q_flow.shape[:-2])   # b,c,t

        if hasattr(self, 'trans_rgb'):
            x_q = self.trans_rgb(x_q)
        x_q_flow = self.trans_flow(x_q_flow)
        x_q = nn.functional.normalize(x_q, dim=1)
        x_q_flow = nn.functional.normalize(x_q_flow, dim=1)
        sim = torch.bmm(x_q.transpose(1, 2), x_q_flow)  # b,t,t

        pos_scores = sim.flatten(0, 1)/self.T
        pos_labels = self.labels.repeat((x_q.shape[0], 1)).flatten(0, 1)
        return dict(pos_scores=pos_scores, pos_labels=pos_labels)

    def extract_global_feat(x):
        pass

    def loss_mx(self, pos_scores, pos_labels, **kwargs):
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

    def loss(self, cls_score, labels, basename=None, **kwargs):
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


@HEADS.register_module()
class MoDistMSEPredHead(MoDistPredHead):
    def __init__(
        self, basename='', loss_cls=dict(type="CrossEntropyLoss"),
        loss_pos=dict(type="CrossEntropyLoss"),
        num_classes=2, in_channels=128, bkb_channels=(512, 128), t=8, T=0.07, pred_weights=(1, 1),
    ):
        super().__init__(basename, loss_cls, loss_pos, num_classes, in_channels, bkb_channels, t, T)
        self.loss_pred = nn.MSELoss()
        self.pred_weights = pred_weights
    
    def forward(self, q_mlvl, q_flow_mlvl, **kwargs):
        x_q = q_mlvl[0]
        x_q_flow = q_flow_mlvl[-1]

        x_q = self.rgb_pooling(x_q).view(*x_q.shape[:-2])   # b,c,t
        x_q_flow = self.flow_pooling(x_q_flow).view(*x_q_flow.shape[:-2])   # b,c,t

        if hasattr(self, 'trans_rgb'):
            x_q = self.trans_rgb(x_q)
        x_q_flow = self.trans_flow(x_q_flow)
        x_q = nn.functional.normalize(x_q, dim=1)
        x_q_flow = nn.functional.normalize(x_q_flow, dim=1)
        sim = torch.bmm(x_q.transpose(1, 2), x_q_flow)  # b,t,t

        pos_scores = sim.flatten(0, 1)/self.T
        pos_labels = self.labels.repeat((x_q.shape[0], 1)).flatten(0, 1)
        return dict(pos_scores=pos_scores, pos_labels=pos_labels, pred_rgb=x_q, pred_flow=x_q_flow)

    def loss_mx(self, pos_scores, pos_labels, pred_rgb, pred_flow, **kwargs):
        # loss for postion prediction
        losses = dict()
        losses['loss_pos'] = self.loss_pos(pos_scores, pos_labels)*self.pred_weights[0]
        losses['loss_pred'] = self.loss_pred(pred_rgb, pred_flow)*self.pred_weights[1] 
        top_k_acc = top_k_accuracy(pos_scores.detach().cpu().numpy(),
                                    pos_labels.detach().cpu().numpy(), (1, 5))
        losses['top1_acc_pos'] = torch.tensor(
            top_k_acc[0], device=pos_scores.device)
        losses['top5_acc_pos'] = torch.tensor(
            top_k_acc[1], device=pos_scores.device)
        return losses


@HEADS.register_module()
class FGMoDistPredHead(MoDistPredHead):
    def __init__(
        self, basename='', loss_cls=dict(type="CrossEntropyLoss"),
        loss_pos=dict(type="CrossEntropyLoss"),
        num_classes=2, in_channels=128, bkb_channels=(512, 128), t=8, T=0.07,
    ):
        super().__init__(basename, loss_cls, loss_pos, num_classes, in_channels, bkb_channels, t, T)
        self.rgb_pooling = nn.AvgPool3d(kernel_size=(1, 4, 4), stride=(1, 4, 4))  # 28 -> 7
        self.flow_pooling = nn.Identity()
        self.trans_flow = nn.Conv3d(bkb_channels[1], 128, 1)
    
    def forward(self, q_mlvl, q_flow_mlvl, **kwargs):
        x_q = q_mlvl[0]
        x_q_flow = q_flow_mlvl[-1]

        x_q = self.rgb_pooling(x_q)   # b,c,t,h,w
        x_q_flow = self.flow_pooling(x_q_flow)
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
        return dict(pos_scores=pos_scores, pos_labels=pos_labels)
        

@HEADS.register_module()
class MoDistPredDTHead(BaseHead):
    """ Head for MoDistPred
    """

    def __init__(
        self, basename='', loss_cls=dict(type="CrossEntropyLoss"),
        loss_pos=dict(type="CrossEntropyLoss"),
        num_classes=2, in_channels=128, bkb_channels=(512, 128), t=8, T=0.07, dth=True,
    ):
        super().__init__(loss_cls=loss_cls, num_classes=num_classes, in_channels=in_channels)
        if basename:
            basename = '_' + basename
        self.loss_pos = build_loss(loss_pos)
        self.basename = basename
        self.rgb_pooling = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.flow_pooling = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.T = T

        if bkb_channels[0] is not None:
            self.trans_rgb = nn.Conv1d(bkb_channels[0], 128, 1)
        else:
            self.trans_rgb = nn.Identity()
        if dth:
            self.trans_flow = lambda x: x.detach()
        else: 
            self.trans_flow = nn.Identity()
        labels = torch.arange(t).unsqueeze(0)
        self.register_buffer('labels', labels)


    def init_weights(self):
        """Initiate the parameters from scratch."""
        pass

    def forward(self, q_mlvl, q_flow_mlvl, **kwargs):
        x_q = q_mlvl[0]
        x_q_flow = q_flow_mlvl[-1]

        x_q = self.rgb_pooling(x_q).view(*x_q.shape[:-2])   # b,c,t
        x_q_flow = self.flow_pooling(x_q_flow).view(*x_q_flow.shape[:-2])   # b,c,t

        if hasattr(self, 'trans_rgb'):
            x_q = self.trans_rgb(x_q)
        x_q_flow = self.trans_flow(x_q_flow)
        x_q = nn.functional.normalize(x_q, dim=1)
        x_q_flow = nn.functional.normalize(x_q_flow, dim=1)
        sim = torch.bmm(x_q.transpose(1, 2), x_q_flow)  # b,t,t

        pos_scores = sim.flatten(0, 1)/self.T
        pos_labels = self.labels.repeat((x_q.shape[0], 1)).flatten(0, 1)
        return dict(pos_scores=pos_scores, pos_labels=pos_labels)

    def extract_global_feat(x):
        pass

    def loss_mx(self, pos_scores, pos_labels, **kwargs):
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

    def loss(self, cls_score, labels, basename=None, **kwargs):
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


@HEADS.register_module()
class MTMoDistPredHead(MoDistPredHead):
    def __init__(
        self, basename='', loss_cls=dict(type="CrossEntropyLoss"),
        loss_pos=dict(type="CrossEntropyLoss"),
        num_classes=2, in_channels=128, bkb_channels=(512, 128), t=8, T=0.07,
    ):
        super().__init__(basename, loss_cls, loss_pos, num_classes, in_channels, bkb_channels, t, T)

    def forward(self, q_mlvl, q_flow_mlvl, **kwargs):
        x_q = q_mlvl[0]
        x_q_flow = q_flow_mlvl[0]

        x_q = self.rgb_pooling(x_q).view(*x_q.shape[:-2])   # b,c,t
        x_q_flow = self.flow_pooling(x_q_flow).view(*x_q_flow.shape[:-2])   # b,c,t

        if hasattr(self, 'trans_rgb'):
            x_q = self.trans_rgb(x_q)
        x_q_flow = self.trans_flow(x_q_flow)
        x_q = nn.functional.normalize(x_q, dim=1)
        x_q_flow = nn.functional.normalize(x_q_flow, dim=1)
        sim = torch.bmm(x_q.transpose(1, 2), x_q_flow)  # b,t,t

        pos_scores = sim.flatten(0, 1)/self.T
        pos_labels = self.labels.repeat((x_q.shape[0], 1)).flatten(0, 1)
        return dict(pos_scores=pos_scores, pos_labels=pos_labels)