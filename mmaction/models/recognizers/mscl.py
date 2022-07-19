import torch
from torch import nn as nn
from torch.nn import functional as F

from ..builder import RECOGNIZERS, build_ssl_aug, build_recognizer
from .base_moco import BaseMoCoRecognizer


@RECOGNIZERS.register_module()
class MSCL(BaseMoCoRecognizer):
    """
    MSCL

    Args:
        recognizer (dict): Moco recognizer for rgb branch.
        recognizer_flow (dict): Moco recognizer for flow branch.
        moco_mx_head (dict): MoCo head for cross-modality learning.
        sup_head (dict): MoCo head for LMCL.
        im_key (str): Key for rgb data in data_batch.
        flow_key (str): Key for flow data in data_batch (e.g. original or visualized flow).
        flow_img_key (str): Key for visualized flow data in aux_info.
        aux_info (list): List of keys, which will also be loaded from data_batch.
        aug (dict): Data augmentation for rgb and flow data.
        samekn (bool): Whether the modality of pos key and negative key are same.
        update_aug_flow (bool): Not used.
        weight_aug_flow (bool): Not used.
        train_cfg (dict | None): Config for training. Default: None.
        test_cfg (dict | None): Config for testing. Default: None.
    """

    def __init__(
        self, recognizer, recognizer_flow, moco_mx_head, sup_head,
        im_key='imgs', flow_key='flows', flow_img_key='flow_imgs', aux_info=[],
        aug=dict(dtype="MoCoAugmentV3", moco_aug=(112, 112), t=8),
        same_kn=True, update_aug_flow=False, weight_aug_flow=(1.0, 1.0),
        train_cfg=None, test_cfg=None,
    ):
        # weight_aug_flow: 0 for intra-modality, 1 for inter modality
        super().__init__(train_cfg=train_cfg, test_cfg=test_cfg)
        self.recognizer = build_recognizer(recognizer)
        self.recognizer_flow = build_recognizer(recognizer_flow)

        self.im_key = im_key
        self.same_kn = same_kn
        self.update_aug_flow = update_aug_flow
        self.weight_aug_flow = weight_aug_flow
        self.flow_key = flow_key
        self.flow_img_key = flow_img_key

        self.aux_info = aux_info
        self._build_cls_head(moco_mx_head, name="moco_mx_head")
        self._build_cls_head(sup_head, name="sup_head")
        self.aug_gpu = build_ssl_aug(aug)

    def train_step(self, data_batch, optimizer, **kwargs):
        im_q = data_batch[self.im_key][0]  # n,c,t,h,w
        im_k = data_batch[self.im_key][1]

        aux_info = {}
        # parse flow key
        aux_info[f'{self.flow_key}_q'] = data_batch[self.flow_key][0]
        aux_info[f'{self.flow_key}_k'] = data_batch[self.flow_key][1]

        for item in self.aux_info:
            assert item in data_batch
            aux_info[item] = data_batch[item]

        batch_size = im_q.shape[0]
        losses = self(im_q, im_k, aux_info, return_loss=True)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(num_samples=batch_size, loss=loss, log_vars=log_vars,)
        return outputs

    def forward(
        self, im_q, im_k, aux_info, return_loss=True, **kwargs):
        # Combination of forward and train/val_step in mmaction.
        if kwargs.get('gradcam', False):
            del kwargs['gradcam']
            return self.forward_gradcam(im_q, im_k, aux_info, **kwargs)
        if return_loss:
            return self.forward_train(im_q, im_k, aux_info, **kwargs)
        else:
            raise NotImplementedError("MoCo doesnt support test mode")

    def forward_train(self, im_q, im_k, aux_info):
        # Before here, random resize_crop and random h_flip is applied.
        im_q, im_k, aux_info = self.aug_gpu(im_q, im_k, aux_info)
        loss_img, im_features = self.recognizer.forward_train(im_q, im_k, aux_info, return_features=True) 
        
        flow_im_q, flow_im_k = aux_info[f'{self.flow_img_key}_q'], aux_info[f'{self.flow_img_key}_k']
        loss_flow, flow_features = self.recognizer_flow.forward_train(flow_im_q, flow_im_k, aux_info, return_features=True) 

        weight = self.recognizer.weight
        weight_flow = self.recognizer_flow.weight

        # q, q_mlvl, k, k_mlvl
        q, k = im_features['q'], im_features['k']
        q_flow, k_flow = flow_features['q'], flow_features['k']
        
        # distill loss
        rf_logits, fr_logits, ssl_label = self.moco_mx_head._forward_moco_mx(q, k, q_flow, k_flow, weight, weight_flow)
        loss_mx = self.moco_mx_head.loss(rf_logits, fr_logits, ssl_label)

        # sup loss
        aux_info = self.sup_head.update_aux_info("im_features", im_features, aux_info)
        aux_info = self.sup_head.update_aux_info("base_flow_features", flow_features, aux_info)
        aux_info_sup = self.sup_head(**aux_info)
        aux_info.update(aux_info_sup)
        loss_sup = self.sup_head.loss(**aux_info)
        # Collect losses
        losses = dict()
        losses.update(loss_img)
        losses.update(loss_flow)
        losses.update(loss_mx)
        losses.update(loss_sup)

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


@RECOGNIZERS.register_module()
class MSCLWithAug(BaseMoCoRecognizer):
    """
    MSCL with flow augmentation

    Args:
        recognizer (dict): Moco recognizer for rgb branch.
        recognizer_flow (dict): Moco recognizer for flow branch.
        moco_mx_head (dict): MoCo head for cross-modality learning.
        sup_head (dict): MoCo head for LMCL.
        im_key (str): Key for rgb data in data_batch.
        flow_key (str): Key for flow data in data_batch (e.g. original or visualized flow).
        aux_info (list): List of keys, which will also be loaded from data_batch.
        aug (dict): Data augmentation for rgb and flow data.
        samekn (bool): Whether the modality of pos key and negative key are same.
        update_aug_flow (bool): Whether to append augmented flow to memory bank.
        weight_aug_flow (bool): Weight to balance different flows.
        train_cfg (dict | None): Config for training. Default: None.
        test_cfg (dict | None): Config for testing. Default: None.
    """

    def __init__(
        self, recognizer, recognizer_flow, moco_mx_head, sup_head,
        im_key='imgs', flow_key='flow_imgs', aux_info=[],
        aug=dict(dtype="MoCoAugmentV3", moco_aug=(112, 112), t=8),
        same_kn=True, update_aug_flow=False, weight_aug_flow=(1.0, 1.0),
        train_cfg=None, test_cfg=None,
    ):
        """
            Args:
                aux_info (tuple, list): keys of aux_info
                aug (dict): parameters for augment builder
                weight_aug_flow (tuple, list): 0 for intra-modality, 1 for inter modality

        """
        super().__init__(train_cfg=train_cfg, test_cfg=test_cfg)
        self.recognizer = build_recognizer(recognizer)
        self.recognizer_flow = build_recognizer(recognizer_flow)

        self.im_key = im_key
        self.same_kn = same_kn
        self.update_aug_flow = update_aug_flow
        self.weight_aug_flow = weight_aug_flow
        if isinstance(flow_key, (list, tuple)):
            self.cat_flow = False
            self.flow_key = flow_key
        else:
            self.cat_flow = True
            self.flow_key = (flow_key, )

        self.aux_info = aux_info
        self._build_cls_head(moco_mx_head, name="moco_mx_head")
        self._build_cls_head(sup_head, name="sup_head")
        self.aug_gpu = build_ssl_aug(aug)

    def train_step(self, data_batch, optimizer, **kwargs):
        im_q = data_batch[self.im_key][0]  # n,c,t,h,w
        im_k = data_batch[self.im_key][1]

        aux_info = {}
        # parse flow key
        for flow_key in self.flow_key:
            aux_info[f'{flow_key}_q'] = data_batch[flow_key][0]
            aux_info[f'{flow_key}_k'] = data_batch[flow_key][1]

        for item in self.aux_info:
            assert item in data_batch
            aux_info[item] = data_batch[item]

        batch_size = im_q.shape[0]
        losses = self(im_q, im_k, aux_info, return_loss=True)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(num_samples=batch_size, loss=loss, log_vars=log_vars,)
        return outputs

    def forward(
        self, im_q, im_k, aux_info, return_loss=True, **kwargs):
        # Combination of forward and train/val_step in mmaction.
        if kwargs.get('gradcam', False):
            del kwargs['gradcam']
            return self.forward_gradcam(im_q, im_k, aux_info, **kwargs)
        if return_loss:
            return self.forward_train(im_q, im_k, aux_info, **kwargs)
        else:
            raise NotImplementedError("MoCo doesnt support test mode")

    def forward_train(self, im_q, im_k, aux_info):
        # Before here, random resize_crop and random h_flip is applied.
        im_q, im_k, aux_info = self.aug_gpu(im_q, im_k, aux_info)
        loss_img, im_features = self.recognizer.forward_train(im_q, im_k, aux_info, return_features=True) 
        
        if self.cat_flow:
            cat_flow_im_q, cat_flow_im_k = aux_info[f'{self.flow_key[0]}_q'], aux_info[f'{self.flow_key[0]}_k']
            flow_im_q, aug_flow_im_q = cat_flow_im_q.chunk(2, 2)    # to 2 chunks in dim 2
            flow_im_k, aug_flow_im_k = cat_flow_im_k.chunk(2, 2)
            flow_im_q, flow_im_k = flow_im_q.contiguous(), flow_im_k.contiguous()
            aug_flow_im_q, aug_flow_im_k = aug_flow_im_q.contiguous(), aug_flow_im_k.contiguous()
        else:
            flow_im_q, flow_im_k = aux_info[f'{self.flow_key[0]}_q'], aux_info[f'{self.flow_key[0]}_k']
            aug_flow_im_q, aug_flow_im_k = aux_info[f'{self.flow_key[1]}_q'], aux_info[f'{self.flow_key[1]}_k']
        loss_base_flow, base_flow_features = self.recognizer_flow.forward_train(flow_im_q, flow_im_k, aux_info, return_features=True) 
        loss_aug_flow, aug_flow_features = self.recognizer_flow.forward_train(aug_flow_im_q, aug_flow_im_k, aux_info, return_features=True, update_queue=self.update_aug_flow) 
        loss_flow = loss_base_flow
        for k in loss_aug_flow:
            if k.startswith('loss'):
                assert k in loss_flow, f"{k} should appear in {loss_flow.keys()}"
                loss_flow[k+'_aug'] = loss_aug_flow[k]*self.weight_aug_flow[0]

        weight = self.recognizer.weight
        weight_flow = self.recognizer_flow.weight

        # q, q_mlvl, k, k_mlvl
        q, k = im_features['q'], im_features['k']
        q_base_flow, k_base_flow = base_flow_features['q'], base_flow_features['k']
        
        # distill loss
        rf_logits, fr_logits, ssl_label = self.moco_mx_head._forward_moco_mx(q, k, q_base_flow, k_base_flow, weight, weight_flow)
        loss_mx = self.moco_mx_head.loss(rf_logits, fr_logits, ssl_label)
        if self.weight_aug_flow[1] > 0:
            q_aug_flow, k_aug_flow = aug_flow_features['q'], aug_flow_features['k']
            aug_rf_logits, aug_fr_logits, aug_ssl_label = self.moco_mx_head._forward_moco_mx(q, k, q_aug_flow, k_aug_flow, weight, weight_flow)
            loss_aug_mx = self.moco_mx_head.loss(aug_rf_logits, aug_fr_logits, aug_ssl_label, suffix='_aug')
            loss_mx.update(loss_aug_mx)

        # sup loss
        aux_info = self.sup_head.update_aux_info("im_features", im_features, aux_info)
        aux_info = self.sup_head.update_aux_info("base_flow_features", base_flow_features, aux_info)
        aux_info = self.sup_head.update_aux_info("aug_flow_features", aug_flow_features, aux_info)
        aux_info_sup = self.sup_head(**aux_info)
        aux_info.update(aux_info_sup)
        loss_sup = self.sup_head.loss(**aux_info)
        # Collect losses
        losses = dict()
        losses.update(loss_img)
        losses.update(loss_flow)
        losses.update(loss_mx)
        losses.update(loss_sup)

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