import torch
from torch import nn as nn

from ..builder import HEADS
from .base import BaseHead
from ...core import top_k_accuracy


@HEADS.register_module()
class RcMoDistHead(BaseHead):
    """ Head for MoDistPred
    """

    def __init__(
        self, basename='', loss_cls=dict(type="CrossEntropyLoss"),
        num_classes=2, in_channels=128, dim_fpn=128,
    ):
        super().__init__(loss_cls=loss_cls, num_classes=num_classes, in_channels=in_channels)
        if basename:
            basename = '_' + basename
        self.loss_rc = nn.MSELoss()
        self.basename = basename

        self.pool2res3 = nn.AvgPool3d((1, 4, 4), stride=(1, 4, 4))
        self.pool_after = nn.AvgPool3d((1, 2, 2), stride=(1, 2, 2))
        self.toimg = nn.Sequential(nn.Conv3d(dim_fpn, 6, 1), nn.Sigmoid())

    def init_weights(self):
        """Initiate the parameters from scratch."""
        pass

    def forward(self, q_flow_mlvl, flow_q, **kwargs):
        flow_q = flow_q.unflatten(2, (flow_q.shape[2]//2, 2))
        flow_q = flow_q.transpose(2, 3).flatten(1, 2)

        rc_loss = 0
        flow_q = self.pool2res3(flow_q)
        for i, ft_fpn in enumerate(q_flow_mlvl):
            pred = self.toimg(ft_fpn)
            rc_loss += self.loss_rc(pred, flow_q)
            if i != len(q_flow_mlvl) - 1:
                flow_q = self.pool_after(flow_q)

        return dict(rc_loss=rc_loss)

    def extract_global_feat(x):
        pass

    def loss_mx(self, rc_loss, **kwargs):
        # loss for postion prediction
        return dict(loss_rc=rc_loss)

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