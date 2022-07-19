from functools import partial

import torch
from torch import nn as nn
from torch.nn import functional as F
from math import pi, cos

from ..builder import RECOGNIZERS, build_ssl_aug
from .base_moco import BaseMoCoRecognizer


def forward(self, x):
    x = self.stem(x)

    out = []
    x = self.layer1(x)
    out.append(x)
    x = self.layer2(x)
    out.append(x)
    x = self.layer3(x)
    out.append(x)
    x = self.layer4(x)
    out.append(x)
    return out


def Identity(*args):
    return args


@RECOGNIZERS.register_module()
class MoCo(BaseMoCoRecognizer):
    """
    MoCo class
    TODO: Adjust to BaseRecognizer style in mmaction2.
    """

    def __init__(
        self, backbone, neck, moco_head, 
        im_key="imgs", dim_in=512, dim=128, K=65536, m=0.999, 
        T=0.07, mlp=False, aux_info=[],
        aug=dict(dtype="MoCoAugmentV3", moco_aug=(112, 112), t=8),
        train_cfg=None, test_cfg=None,
    ):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super().__init__(train_cfg=train_cfg, test_cfg=test_cfg)

        self.K = K
        self.m = m
        self.T = T
        self.im_key = im_key
        self.aux_info = aux_info

        # create the encoders
        # num_classes is the output fc dimension
        # config: dict(type="torchvision.r3d_18", num_classes=128)
        if "num_classes" in backbone:
            assert backbone["num_classes"] == dim
        backbone_k = backbone.copy()  # 浅拷贝即可
        self._build_backbone(backbone, "encoder_q")
        self._build_backbone(backbone_k, "encoder_k")
        # * For simplify, neck must be used(can be identity)
        self._build_neck(neck, "neck_q")
        self._build_neck(neck, "neck_k")
        # * This can be either cls_head or roi_head
        self._build_cls_head(moco_head, "moco_head")
        self.init_weights()

        # assert mlp, "Please align with ReSim, then delete this code..."
        if mlp:
            self.mlp_q = nn.Sequential(nn.Linear(dim_in, dim_in), nn.ReLU(), nn.Linear(dim_in, dim))
            self.mlp_k = nn.Sequential(nn.Linear(dim_in, dim_in), nn.ReLU(), nn.Linear(dim_in, dim))
        else:
            self.mlp_q = nn.Linear(dim_in, dim)
            self.mlp_k = nn.Linear(dim_in, dim)

        if self.backbone_from == "torchvision":
            self.encoder_q.forward = partial(forward, self.encoder_q)
            self.encoder_k.forward = partial(forward, self.encoder_k)
            

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for param_q, param_k in zip(self.neck_q.parameters(), self.neck_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for param_q, param_k in zip(self.mlp_q.parameters(), self.mlp_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # self.count = torch.zeros(K, dtype=torch.int16).cuda()
        self.register_buffer("count", torch.zeros(K, dtype=torch.long))
        self._weight = None

        # create cls head, for simplify, simple loss in pytorch is
        # TODO: Add cls_head in mmaction2.
        # self.cls_head = nn.CrossEntropyLoss()

        # create augment
        self.aug_gpu = build_ssl_aug(aug)

        # Parse 

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
        for param_q, param_k in zip(self.neck_q.parameters(), self.neck_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
        for param_q, param_k in zip(self.mlp_q.parameters(), self.mlp_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        self.count += 1
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T

        self.count[ptr : ptr + batch_size] = 1
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda(non_blocking=True)

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def train_step(self, data_batch, optimizer, **kwargs):
        # optimizer for GAN-style training
        # for both train_step and val_step, as this is the same process now!

        im_q = data_batch[self.im_key][0]  # n,c,t,h,w
        im_k = data_batch[self.im_key][1]

        aux_info = {}
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

    def forward_train(self, im_q, im_k, aux_info, return_features=False, update_queue=True):
        # im_q, im_k = imgs
        # outputs = dict()

        # Before here, random resize_crop and random h_flip is applied.
        # For simplify, drop aug when return_feature=True 
        if return_features:
            aux_info = aux_info.copy()  # Not change the key of aux_info
        else:
            im_q, im_k, aux_info = self.aug_gpu(im_q, im_k, aux_info)   # aug for single training
        q, q_mlvl, k, k_mlvl, sup_loss = self.extract_feat(im_q, im_k)

        # === INSTANCE ===
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        # temporal decay: t = 0.99999
        weight = 0.99999 ** (1.0 * self.count).cuda()
        weight = torch.mul(self.queue, weight).cuda()
        weight = weight.clone().detach()
        # negative sample in current iter
        self._weight = weight
        l_neg = torch.einsum("nc,ck->nk", [q, weight])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        ssl_label = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        if update_queue:
            self._dequeue_and_enqueue(k)
        aux_info['q_mlvl'] = q_mlvl
        aux_info['k_mlvl'] = k_mlvl
        losses = self.moco_head.loss(logits, ssl_label, **aux_info)
        losses.update(sup_loss)

        if return_features:
            return losses, dict(q=q, q_mlvl=q_mlvl, k=k, k_mlvl=k_mlvl, q_neg=l_neg)
        else:
            return losses

    def forward_test(self, imgs):
        raise NotImplementedError("TODO")

    def forward_gradcam(self, imgs):
        raise NotImplementedError("TODO")

    def extract_global_feat(self):
        raise NotImplementedError("TODO")

    def extract_feat(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        # compute query features
        q_mlvl = self.encoder_q(im_q)
        (q_emb, q_mlvl), sup_loss = self.neck_q(q_mlvl)
        q = self.mlp_q(q_emb)
        q = nn.functional.normalize(q, dim=1)  # NxC

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k_mlvl = self.encoder_k(im_k)
            (k_emb, k_mlvl), _ = self.neck_k(k_mlvl)  # keys: NxC
            k = self.mlp_k(k_emb)
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            k_mlvl = [self._batch_unshuffle_ddp(k_slvl, idx_unshuffle) for k_slvl in k_mlvl]

        return q, q_mlvl, k, k_mlvl, sup_loss

    @property
    def weight(self):
        return self._weight

    def visualize(self, data_batch):
        pass


@RECOGNIZERS.register_module()
class MoCoV2(MoCo):
    """
    MoCo class and support momentum annealing for m
    """

    def __init__(
        self, backbone, neck, moco_head, 
        im_key="imgs", dim_in=512, dim=128, K=65536, m_base=0.994, t_decay=0.99999, max_iters=1,
        T=0.07, mlp=False, aux_info=[],
        aug=dict(dtype="MoCoAugmentV3", moco_aug=(112, 112), t=8),
        train_cfg=None, test_cfg=None,
    ):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__(train_cfg=train_cfg, test_cfg=test_cfg)

        self.K = K
        self.m_base = m_base
        self.m = m_base
        self.iters = 0
        self.max_iters = max_iters
        self.batch_size = 0
        self.T = T

        self.im_key = im_key
        self.t_decay = t_decay
        self.aux_info = aux_info

        # create the encoders
        # num_classes is the output fc dimension
        # config: dict(type="torchvision.r3d_18", num_classes=128)
        if "num_classes" in backbone:
            assert backbone["num_classes"] == dim
        backbone_k = backbone.copy()  # 浅拷贝即可
        self._build_backbone(backbone, "encoder_q")
        self._build_backbone(backbone_k, "encoder_k")
        # * For simplify, neck must be used(can be identity)
        self._build_neck(neck, "neck_q")
        self._build_neck(neck, "neck_k")
        # * This can be either cls_head or roi_head
        self._build_cls_head(moco_head, "moco_head")
        self.init_weights()

        # assert mlp, "Please align with ReSim, then delete this code..."
        if mlp:
            self.mlp_q = nn.Sequential(nn.Linear(dim_in, dim_in), nn.ReLU(), nn.Linear(dim_in, dim))
            self.mlp_k = nn.Sequential(nn.Linear(dim_in, dim_in), nn.ReLU(), nn.Linear(dim_in, dim))
        else:
            self.mlp_q = nn.Linear(dim_in, dim)
            self.mlp_k = nn.Linear(dim_in, dim)

        if self.backbone_from == "torchvision":
            self.encoder_q.forward = partial(forward, self.encoder_q)
            self.encoder_k.forward = partial(forward, self.encoder_k)
            

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for param_q, param_k in zip(self.neck_q.parameters(), self.neck_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for param_q, param_k in zip(self.mlp_q.parameters(), self.mlp_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # self.count = torch.zeros(K, dtype=torch.int16).cuda()
        self.register_buffer("count", torch.zeros(K, dtype=torch.long))
        self._weight = None

        # create cls head, for simplify, simple loss in pytorch is
        # TODO: Add cls_head in mmaction2.
        # self.cls_head = nn.CrossEntropyLoss()

        # create augment
        self.aug_gpu = build_ssl_aug(aug)

        # Parse 

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        factor = min(self.iters/self.max_iters, 1)
        weight = 1-0.5*(1-self.m_base)*(cos(pi*factor)+1)
        self.m = 1*weight
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
        for param_q, param_k in zip(self.neck_q.parameters(), self.neck_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
        for param_q, param_k in zip(self.mlp_q.parameters(), self.mlp_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        self.count += 1
        batch_size = keys.shape[0]
        self.batch_size = batch_size

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T

        self.count[ptr : ptr + batch_size] = 1
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def train_step(self, data_batch, optimizer, **kwargs):
        # optimizer for GAN-style training
        # for both train_step and val_step, as this is the same process now!

        im_q = data_batch[self.im_key][0]  # n,c,t,h,w
        im_k = data_batch[self.im_key][1]

        aux_info = {}
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

    def forward_train(self, im_q, im_k, aux_info, return_features=False, update_queue=True):
        if return_features:
            aux_info = aux_info.copy()  # Not change the key of aux_info
        else:
            im_q, im_k, aux_info = self.aug_gpu(im_q, im_k, aux_info)   # aug for single training
        q, q_mlvl, k, k_mlvl, sup_loss = self.extract_feat(im_q, im_k)

        # === INSTANCE ===
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        # temporal decay: t = 0.99999
        weight = 0.99999 ** (1.0 * self.count).cuda()
        weight = torch.mul(self.queue, weight).cuda()
        weight = weight.clone().detach()
        # negative sample in current iter
        self._weight = weight
        l_neg = torch.einsum("nc,ck->nk", [q, weight])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        ssl_label = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        if update_queue:
            self._dequeue_and_enqueue(k)
        # update iters
        if self.training:
            self.iters += self.batch_size
        
        aux_info['q_mlvl'] = q_mlvl
        aux_info['k_mlvl'] = k_mlvl
        losses = self.moco_head.loss(logits, ssl_label, **aux_info)
        losses.update(sup_loss)

        if return_features:
            return losses, dict(q=q, q_mlvl=q_mlvl, k=k, k_mlvl=k_mlvl, q_neg=l_neg)
        else:
            return losses

    def extract_feat(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        # compute query features
        q_mlvl = self.encoder_q(im_q)
        (q_emb, q_mlvl), sup_loss = self.neck_q(q_mlvl)
        q = self.mlp_q(q_emb)
        q = nn.functional.normalize(q, dim=1)  # NxC

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k_mlvl = self.encoder_k(im_k)
            (k_emb, k_mlvl), _ = self.neck_k(k_mlvl)  # keys: NxC
            k = self.mlp_k(k_emb)
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            k_mlvl = [self._batch_unshuffle_ddp(k_slvl, idx_unshuffle) for k_slvl in k_mlvl]

        return q, q_mlvl, k, k_mlvl, sup_loss

    @property
    def weight(self):
        return self._weight

    def visualize(self, data_batch):
        pass


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output