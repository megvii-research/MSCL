import kornia
import torch
import torch.nn as nn
from torchvision import transforms as transforms

from ..builder import SSL_AUGS
from .ssl_aug import (GaussianBlur, __video_batch_prob_generator__,
                      VideoRandomApply, toVideoAug, toConsistentAug, FlowVisualizer)

"""
    aug_strong = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    aug_weak = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
"""

def get_aug(sync_op, ts, crop_size, normalize_video, weak=False):
    if weak:
        return normalize_video
    else:
        return transforms.Compose(
                [
                    sync_op(kornia.augmentation.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8), ts),
                    sync_op(kornia.augmentation.RandomGrayscale(p=0.2), ts),
                    VideoRandomApply(GaussianBlur([0.1, 2.0], crop_size), ts, p=0.5),
                    normalize_video,
                ])

def get_sync_op(sync_level):
    if sync_level == "batch":
        sync_op = toVideoAug
    elif sync_level == "params":
        sync_op = toConsistentAug
    return sync_op

@SSL_AUGS.register_module()
class SyncMoCoAugmentV5(object):
    """
    V5: similar to V3, but add some new parameters:
    new parameters:
    - weak_aug, whether to use weak_aug for query and key, default: (False, False)
    - normalize, whether to normalize flow, default: False
    """
    def __init__(
        self, crop_size, flip_transform=dict(p=0.5, same_on_batch=False),
        sync_level="batch", t=None, flow_suffix='flow_imgs', img_width=112,
        visualize=True, weak_aug=(False, False), normalize_flow=False,
    ):
        if isinstance(sync_level, str):
            sync_level = (sync_level, sync_level)
        assert all((v in ["batch", "params"] for v in sync_level))
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        normalize_video = kornia.augmentation.Normalize(mean, std)
        # ColorJitter是batch-level的，RandomGrayscale是image-level的。

        if visualize:
            self.visualizer = FlowVisualizer()
        else:
            self.visualizer = nn.Identity()
        
        self.flow_suffix = flow_suffix
        self.img_width = img_width
        if flow_suffix is not None:
            ts = t[0]
        else:
            ts = t

        if flip_transform:
            self.flip_transform = kornia.augmentation.augmentation3d.RandomHorizontalFlip3D(**flip_transform)
        else:
            self.flip_transform = None
        self.moco_augment = list((get_aug(get_sync_op(sync_level[cid]), ts, crop_size, normalize_video, weak_aug[cid]) for cid in range(2)))
        self.flow_normalizer = normalize_video if normalize_flow else nn.Identity()

    def __call__(self, im_q, im_k, aux_info):
        im_q, aux_info = self.forward_flip(im_q, aux_info, suffix='_q')
        im_q = self.single_cal(im_q, func=self.moco_augment[0])

        im_k, aux_info = self.forward_flip(im_k, aux_info, suffix='_k')
        im_k = self.single_cal(im_k, func=self.moco_augment[1])

        return im_q, im_k, aux_info

    def bbox_hflip(self, bboxes):
        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.clone()

        flipped[..., 0::4] = self.img_width - bboxes[..., 2::4]
        flipped[..., 2::4] = self.img_width - bboxes[..., 0::4]
        return flipped

    def forward_flip(self, clips, aux_info, suffix='_q'):
        # (B, C, T, H, W)
        clips = self.flip_transform(clips)
        to_apply = self.flip_transform._params["batch_prob"]
        if self.flow_suffix:
            full_suffix = self.flow_suffix + suffix
            for k in aux_info:
                if k.endswith(full_suffix):
                    aux_info[k] = self.visualizer(aux_info[k])
                    aux_info[k] = self.single_cal(aux_info[k], func=self.flow_normalizer)
                    aux_info[k][to_apply] = torch.flip(aux_info[k][to_apply], [-1])
        if 'gt_bboxes'+suffix in aux_info:
            box_key = 'gt_bboxes'+suffix
            for i, flag in enumerate(to_apply):
                if flag:
                    aux_info[box_key][i] = self.bbox_hflip(aux_info[box_key][i])
        return clips, aux_info

    def single_cal(self, clips, func):
        # from (B, C, T, H, W) to (B, T, C, H, W)
        clips = clips.permute(0, 2, 1, 3, 4).contiguous()
        clips_batch = clips.view(-1, clips.shape[2], clips.shape[3], clips.shape[4])
        aug_clips = func(clips_batch)
        aug_clips = aug_clips.view(clips.shape)
        # from (B, T, C, H, W) to (B, C, T, H, W)
        aug_clips = aug_clips.permute(0, 2, 1, 3, 4).contiguous()
        return aug_clips