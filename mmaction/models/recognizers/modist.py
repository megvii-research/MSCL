import torch
from torch import nn as nn
from torch.nn import functional as F

from ..builder import RECOGNIZERS, build_ssl_aug, build_recognizer
from .base_moco import BaseMoCoRecognizer


@RECOGNIZERS.register_module()
class MoDist(BaseMoCoRecognizer):
    """
    Reimplemented MoDist

    Args:
        recognizer (dict): Moco recognizer for rgb branch.
        recognizer_flow (dict): Moco recognizer for flow branch.
        moco_head (dict): MoCo head for cross-modality learning.
        im_key (str): Key for rgb data in data_batch.
        flow_key (str): Key for flow data in data_batch (e.g. original or visualized flow).
        aux_info (list): List of keys, which will also be loaded from data_batch.
        aug (dict): Data augmentation for rgb and flow data.
        samekn (bool): Whether the modality of pos key and negative key are same.
        train_cfg (dict | None): Config for training. Default: None.
        test_cfg (dict | None): Config for testing. Default: None.
    """

    def __init__(
        self, recognizer, recognizer_flow, moco_head, 
        im_key='imgs', flow_key='flow_imgs', aux_info=[],
        aug=dict(dtype="MoCoAugmentV3", moco_aug=(112, 112), t=8),
        same_kn=True, train_cfg=None, test_cfg=None,
    ):
        super().__init__(train_cfg=train_cfg, test_cfg=test_cfg)
        self.recognizer = build_recognizer(recognizer)
        self.recognizer_flow = build_recognizer(recognizer_flow)
        self.T = self.recognizer.T
        self.im_key = im_key
        self.flow_key = flow_key
        self.same_kn = same_kn

        self.aux_info = aux_info
        moco_head_r = moco_head.copy()
        moco_head_r.basename += '_r'
        self._build_cls_head(moco_head, name="moco_head")
        self._build_cls_head(moco_head_r, name="moco_head_r")
        self.aug_gpu = build_ssl_aug(aug)

    def train_step(self, data_batch, optimizer, **kwargs):
        # optimizer for GAN-style training
        # for both train_step and val_step, as this is the same process now!

        im_q = data_batch[self.im_key][0]  # n,c,t,h,w
        im_k = data_batch[self.im_key][1]

        flow_q = data_batch[self.flow_key][0]  # n,c,t,h,w
        flow_k = data_batch[self.flow_key][1]

        aux_info = {}
        for item in self.aux_info:
            assert item in data_batch
            aux_info[item] = data_batch[item]

        batch_size = im_q.shape[0]
        losses = self(im_q, im_k, flow_q, flow_k, aux_info, return_loss=True)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(num_samples=batch_size, loss=loss, log_vars=log_vars,)
        return outputs

    def forward(
        self, im_q, im_k, flow_q, flow_k, aux_info, return_loss=True, **kwargs):
        # Combination of forward and train/val_step in mmaction.
        if kwargs.get('gradcam', False):
            del kwargs['gradcam']
            return self.forward_gradcam(im_q, im_k, flow_q, flow_k, aux_info, **kwargs)
        if return_loss:
            return self.forward_train(im_q, im_k, flow_q, flow_k, aux_info, **kwargs)
        else:
            raise NotImplementedError("MoCo doesnt support test mode")

    def forward_train(self, im_q, im_k, flow_q, flow_k, aux_info):
        # Before here, random resize_crop and random h_flip is applied.
        im_q, im_k, flow_q, flow_k, aux_info = self.aug_gpu.forward_with_flow(im_q, im_k, flow_q, flow_k, aux_info)
        loss_img, im_features = self.recognizer.forward_train(im_q, im_k, aux_info, return_features=True) 
        loss_flow, flow_features = self.recognizer_flow.forward_train(flow_q, flow_k, aux_info, return_features=True) 

        weight = self.recognizer.weight
        weight_flow = self.recognizer_flow.weight

        # q, q_mlvl, k, k_mlvl
        q, k = im_features['q'], im_features['k']
        q_flow, k_flow = flow_features['q'], flow_features['k']

        rf_l_pos = torch.einsum("nc,nc->n", [q, k_flow]).unsqueeze(-1)
        fr_l_pos = torch.einsum("nc,nc->n", [q_flow, k]).unsqueeze(-1)
        if self.same_kn:
            rf_l_neg = torch.einsum("nc,ck->nk", [q, weight_flow])
            fr_l_neg = torch.einsum("nc,ck->nk", [q_flow, weight])
        else:
            rf_l_neg = torch.einsum("nc,ck->nk", [q, weight])
            fr_l_neg = torch.einsum("nc,ck->nk", [q_flow, weight_flow])
        # print((q*k).sum(1), (q*k_flow).sum(1), (q_flow*k).sum(1), (q_flow*k_flow).sum(1))
        # print("Rev: ", (q@weight).max(1), (q_flow@weight_flow).max(1), (q@weight_flow).max(1), (q_flow@weight).max(1))

        rf_logits = torch.cat([rf_l_pos, rf_l_neg], dim=1)/self.T
        fr_logits = torch.cat([fr_l_pos, fr_l_neg], dim=1)/self.T

        # labels: positive key indicators
        ssl_label = torch.zeros(rf_logits.shape[0], dtype=torch.long).cuda()

        losses = self.moco_head.loss(rf_logits, ssl_label, **aux_info)
        losses_fr = self.moco_head_r.loss(fr_logits, ssl_label, **aux_info)
        losses.update(losses_fr)
        losses.update(loss_img)
        losses.update(loss_flow)

        return losses

    def forward_test(self, imgs):
        raise NotImplementedError("Not support for ssl recognizer !!!")

    def forward_gradcam(self, imgs):
        raise NotImplementedError("Not support for ssl recognizer !!!")

    def extract_global_feat(self):
        raise NotImplementedError("Not support for ssl recognizer !!!")

    def extract_feat(self, im_q, im_k):
        pass

    def visualize(self, data_batch):
        pass