import random
from functools import partial
from typing import Tuple, Union

import kornia
import torch
import math
import torch.nn as nn
from kornia.augmentation.utils import _adapted_sampling
from torch.distributions import Bernoulli
from torch.utils.data import Sampler
from torchvision import transforms as transforms
from torchvision.datasets.video_utils import VideoClips

from tools.RAFT.core.utils.flow_viz import make_colorwheel

from .motion_map_calculator import MotionMapCalculator
from ..builder import SSL_AUGS


def _adapted_sampling_video(
    shape: Union[Tuple, torch.Size], dist: torch.distributions.Distribution, same_on_batch=False
) -> torch.Tensor:
    r"""The uniform sampling function that accepts 'same_on_batch'.

    If same_on_batch is True, all values generated will be exactly same given a batch_size (shape[0]).
    By default, same_on_batch is set to False.
    """
    assert len(shape) == 2, f"{shape}"
    return dist.sample((shape[0], 1)).repeat(1, shape[1]).view(-1)


def __video_batch_prob_generator__(
    batch_shape: torch.Size, p: float, p_batch: float, same_on_batch: bool, self=None, t=None
) -> torch.Tensor:
    batch_prob: torch.Tensor
    bs = batch_shape[0] // t
    if p_batch == 1:
        batch_prob = torch.tensor([True])
    elif p_batch == 0:
        batch_prob = torch.tensor([False])
    else:
        batch_prob = _adapted_sampling((1,), self._p_batch_gen, same_on_batch).bool()

    if batch_prob.sum().item() == 1:
        elem_prob: torch.Tensor
        if p == 1:
            elem_prob = torch.tensor([True] * batch_shape[0])
        elif p == 0:
            elem_prob = torch.tensor([False] * batch_shape[0])
        else:
            elem_prob = _adapted_sampling_video((bs, t), self._p_gen, same_on_batch).bool()
        batch_prob = batch_prob * elem_prob
    else:
        batch_prob = batch_prob.repeat(batch_shape[0])
    return batch_prob

def forward_parameters_video(
    batch_shape: torch.Size, self=None, t=None) -> torch.Tensor:
    bs = batch_shape[0] // t
    to_apply = self.__batch_prob_generator__((bs,), self.p, self.p_batch, self.same_on_batch)
    _params = self.generate_parameters(
        torch.Size((int(to_apply.sum().item()), *batch_shape[1:])))
    if _params is None:
        _params = {}
    _params['batch_prob'] = to_apply
    # Repeat the parameters
    for key in _params:
        if key != 'order':
            new_param = _params[key]   # B, 1, ...
            new_param = new_param.unsqueeze(1).expand(-1, t, *new_param.shape[1:])
            new_param = new_param.flatten(0, 1)
            _params[key] = new_param
    return _params

def toVideoAug(aug, t):
    assert t is not None
    aug.__batch_prob_generator__ = partial(__video_batch_prob_generator__, self=aug, t=t)
    return aug

def toConsistentAug(aug, t):
    # Can be applied to ColorJitter/RandomGrayscale
    assert t is not None
    aug.forward_parameters = partial(forward_parameters_video, self=aug, t=t)
    return aug

def flow_uv_to_colors(u: torch.Tensor, v, colorwheel, convert_to_bgr=False, div255=True):
    """
    Args:
        u (np.ndarray): Input horizontal flow of shape [B,H,W]
        v (np.ndarray): Input vertical flow of shape [B,H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = torch.zeros((u.shape[0], u.shape[1], u.shape[2], 3), dtype=torch.uint8, device=u.device)
    ncols = colorwheel.shape[0]
    rad = torch.sqrt(torch.square(u) + torch.square(v))
    a = torch.atan2(-v, -u)/math.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = torch.floor(fk).long()
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[...,ch_idx] = torch.floor(255 * col) # bt,h,w,c
    flow_image = flow_image.float()
    if div255:
        flow_image /= 255
    return flow_image

class FlowVisualizer:
    def __init__(self) -> None:
        colorwheel = make_colorwheel()
        self.colorwheel = torch.from_numpy(colorwheel)

    def __call__(self, flows):
        # flows -> b,2,t,h,w
        bs, _, t = flows.shape[:3]

        self.colorwheel = self.colorwheel.to(flows.device)
        u, v = flows.chunk(2, dim=1)
        u, v = u[:, 0].flatten(0, 1), v[:, 0].flatten(0, 1)
        flow_imgs = flow_uv_to_colors(u, v, self.colorwheel, convert_to_bgr=False, div255=True) # bt,h,w,c
        flow_imgs = flow_imgs.unflatten(0, (bs, t)).permute(0, 4, 1, 2, 3)
        return flow_imgs

class VideoRandomApply(torch.nn.Module):
    """Apply randomly a list of transformations with a given probability.
    This is consistent across time.
    """

    def __init__(self, transform, t, p=0.5):
        super().__init__()
        self.transform = transform
        self.p = p
        self.t = t
        self._p_gen = Bernoulli(self.p)
        self._p_batch_gen = Bernoulli(1)

    def forward(self, img):
        batch_prob = __video_batch_prob_generator__(img.shape[0:1], self.p, 1, same_on_batch=False, self=self, t=self.t)
        if torch.any(batch_prob):
            img[batch_prob] = self.transform(img[batch_prob])
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += "\n    p={}".format(self.p)
        format_string += "\n"
        format_string += "    {0}".format(self.transform)
        format_string += "\n)"
        return format_string


class GaussianBlur(object):
    def __init__(self, sigma=[0.1, 2.0], img_size=112):
        self.sigma = sigma
        self.radius = int(0.1 * img_size) // 2 * 2 + 1

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        gauss = kornia.filters.GaussianBlur2d((self.radius, self.radius), (sigma, sigma))
        return gauss(x)


@SSL_AUGS.register_module()
class IdentityAug(object):
    def __init__(self):
        pass

    def __call__(self, clips):
        return clips


@SSL_AUGS.register_module()
class MoCoAugment(object):
    def __init__(self, crop_size):

        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        normalize_video = kornia.augmentation.Normalize(mean, std)  # (x-mean)/std
        self.moco_augment = transforms.Compose(
            [
                kornia.augmentation.RandomGrayscale(p=0.2),
                kornia.augmentation.ColorJitter(0.4, 0.4, 0.4, 0.4),
                kornia.augmentation.RandomHorizontalFlip(),  # default parameters(p=0.5, not same on batch).
                normalize_video,
            ]
        )

    def __call__(self, clips):
        # from (B, C, T, H, W) to (B, T, C, H, W)
        # clips need to be float and normalized into [0, 1] for the best differentiability support.
        clips = clips.permute(0, 2, 1, 3, 4).contiguous()
        clips_batch = clips.view(-1, clips.shape[2], clips.shape[3], clips.shape[4])
        aug_clips = self.moco_augment(clips_batch)
        aug_clips = aug_clips.view(clips.shape)
        # from (B, T, C, H, W) to (B, C, T, H, W)
        aug_clips = aug_clips.permute(0, 2, 1, 3, 4).contiguous()
        return aug_clips

@SSL_AUGS.register_module()
class MoCoAugmentV2(object):
    def __init__(self, crop_size):

        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        normalize_video = kornia.augmentation.Normalize(mean, std)
        self.moco_augment_v2 = transforms.Compose(
            [
                transforms.RandomApply(
                    [kornia.augmentation.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
                ),
                kornia.augmentation.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0], crop_size)], p=0.5),
                kornia.augmentation.RandomHorizontalFlip(),
                normalize_video,
            ]
        )

    def __call__(self, im_q, im_k, aux_info):
        im_q = self.single_cal(im_q)
        im_k = self.single_cal(im_k)

        return im_q, im_k, aux_info

    def single_cal(self, clips):
        # from (B, C, T, H, W) to (B, T, C, H, W)
        clips = clips.permute(0, 2, 1, 3, 4).contiguous()
        clips_batch = clips.view(-1, clips.shape[2], clips.shape[3], clips.shape[4])
        aug_clips = self.moco_augment_v2(clips_batch)
        aug_clips = aug_clips.view(clips.shape)
        # from (B, T, C, H, W) to (B, C, T, H, W)
        aug_clips = aug_clips.permute(0, 2, 1, 3, 4).contiguous()
        return aug_clips

@SSL_AUGS.register_module()
class SyncMoCoAugmentV2(object):
    def __init__(
        self, crop_size, flip_transform=dict(p=0.5, same_on_batch=False),
        sync_level="batch", t=None, with_flow=False, img_width=112,
    ):
        assert sync_level in ["batch", "params"]
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        normalize_video = kornia.augmentation.Normalize(mean, std)
        # ColorJitter是batch-level的，RandomGrayscale是image-level的。
        if sync_level == "batch":
            sync_op = toVideoAug
        elif sync_level == "params":
            sync_op = toConsistentAug
        
        self.with_flow = with_flow
        self.img_width = img_width
        if with_flow:
            ts = t[0]
        else:
            ts = t

        if flip_transform:
            self.flip_transform = kornia.augmentation.augmentation3d.RandomHorizontalFlip3D(**flip_transform)
        else:
            self.flip_transform = None
        self.moco_augment_v2 = transforms.Compose(
            [
                sync_op(kornia.augmentation.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8), ts),
                sync_op(kornia.augmentation.RandomGrayscale(p=0.2), ts),
                VideoRandomApply(GaussianBlur([0.1, 2.0], crop_size), ts, p=0.5),
                normalize_video,
            ])

    def __call__(self, im_q, im_k, aux_info):
        im_q, aux_info, _ = self.forward_flip(im_q, aux_info, suffix='_q')
        im_q = self.single_cal(im_q)

        im_k, aux_info, _ = self.forward_flip(im_k, aux_info, suffix='_k')
        im_k = self.single_cal(im_k)

        return im_q, im_k, aux_info

    def bbox_hflip(self, bboxes):
        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.clone()

        flipped[..., 0::4] = self.img_width - bboxes[..., 2::4]
        flipped[..., 2::4] = self.img_width - bboxes[..., 0::4]
        return flipped

    def forward_flip(self, clips, aux_info, clips_flow=None, suffix='_q'):
        # (B, C, T, H, W)
        clips = self.flip_transform(clips)
        to_apply = self.flip_transform._params["batch_prob"]
        if self.with_flow:
            clips_flow[to_apply] = torch.flip(clips_flow[to_apply], [-1])
        if 'gt_bboxes'+suffix in aux_info:
            box_key = 'gt_bboxes'+suffix
            for i, flag in enumerate(to_apply):
                if flag:
                    aux_info[box_key][i] = self.bbox_hflip(aux_info[box_key][i])
        return clips, aux_info, clips_flow

    def forward_with_flow(self, im_q, im_k, flow_q, flow_k, aux_info):
        im_q, aux_info, flow_q = self.forward_flip(im_q, aux_info, flow_q, suffix='_q')
        im_q = self.single_cal(im_q)

        im_k, aux_info, flow_k = self.forward_flip(im_k, aux_info, flow_k, suffix='_k')
        im_k = self.single_cal(im_k)

        return im_q, im_k, flow_q, flow_k, aux_info

    def single_cal(self, clips):
        # from (B, C, T, H, W) to (B, T, C, H, W)
        clips = clips.permute(0, 2, 1, 3, 4).contiguous()
        clips_batch = clips.view(-1, clips.shape[2], clips.shape[3], clips.shape[4])
        aug_clips = self.moco_augment_v2(clips_batch)
        aug_clips = aug_clips.view(clips.shape)
        # from (B, T, C, H, W) to (B, C, T, H, W)
        aug_clips = aug_clips.permute(0, 2, 1, 3, 4).contiguous()
        return aug_clips


@SSL_AUGS.register_module()
class SyncMoCoAugmentV3(object):
    def __init__(
        self, crop_size, flip_transform=dict(p=0.5, same_on_batch=False),
        sync_level="batch", t=None, flow_suffix='flow_imgs', img_width=112,
        visualize=True,
    ):
        assert sync_level in ["batch", "params"]
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        normalize_video = kornia.augmentation.Normalize(mean, std)
        # ColorJitter是batch-level的，RandomGrayscale是image-level的。
        if sync_level == "batch":
            sync_op = toVideoAug
        elif sync_level == "params":
            sync_op = toConsistentAug

        if visualize:
            self.visualizer = FlowVisualizer()
        else:
            self.visualizer = torch.nn.Identity()
        
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
        self.moco_augment_v2 = transforms.Compose(
            [
                sync_op(kornia.augmentation.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8), ts),
                sync_op(kornia.augmentation.RandomGrayscale(p=0.2), ts),
                VideoRandomApply(GaussianBlur([0.1, 2.0], crop_size), ts, p=0.5),
                normalize_video,
            ])

    def __call__(self, im_q, im_k, aux_info):
        im_q, aux_info = self.forward_flip(im_q, aux_info, suffix='_q')
        im_q = self.single_cal(im_q)

        im_k, aux_info = self.forward_flip(im_k, aux_info, suffix='_k')
        im_k = self.single_cal(im_k)

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
                    aux_info[k][to_apply] = torch.flip(aux_info[k][to_apply], [-1])
        if 'gt_bboxes'+suffix in aux_info:
            box_key = 'gt_bboxes'+suffix
            for i, flag in enumerate(to_apply):
                if flag:
                    aux_info[box_key][i] = self.bbox_hflip(aux_info[box_key][i])
        return clips, aux_info

    def single_cal(self, clips):
        # from (B, C, T, H, W) to (B, T, C, H, W)
        clips = clips.permute(0, 2, 1, 3, 4).contiguous()
        clips_batch = clips.view(-1, clips.shape[2], clips.shape[3], clips.shape[4])
        aug_clips = self.moco_augment_v2(clips_batch)
        aug_clips = aug_clips.view(clips.shape)
        # from (B, T, C, H, W) to (B, C, T, H, W)
        aug_clips = aug_clips.permute(0, 2, 1, 3, 4).contiguous()
        return aug_clips


@SSL_AUGS.register_module()
class SyncMoCoAugmentV4(nn.Module):
    """ v4 -> similart to v3, but use flows as input...
    """
    def __init__(
        self, crop_size, flip_transform=dict(p=0.5, same_on_batch=False),
        sync_level="batch", t=None, flow_suffix='flows', img_width=112,
        motion_calculator_params=dict(scales=(7, 7), pool_type='max'), visualize=True,
    ):
        super().__init__()
        assert sync_level in ["batch", "params"]
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        normalize_video = kornia.augmentation.Normalize(mean, std)

        if sync_level == "batch":
            sync_op = toVideoAug
        elif sync_level == "params":
            sync_op = toConsistentAug

        if visualize:
            self.visualizer = FlowVisualizer()
        else:
            self.visualizer = torch.nn.Identity()
        self.motion_calculator = MotionMapCalculator(**motion_calculator_params)
        
        self.flow_suffix = flow_suffix
        assert self.flow_suffix != "flow_imgs"
        self.img_width = img_width
        if flow_suffix is not None:
            ts = t[0]
        else:
            ts = t

        if flip_transform:
            self.flip_transform = kornia.augmentation.augmentation3d.RandomHorizontalFlip3D(**flip_transform)
        else:
            self.flip_transform = None
        self.moco_augment_v2 = transforms.Compose(
            [
                sync_op(kornia.augmentation.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8), ts),
                sync_op(kornia.augmentation.RandomGrayscale(p=0.2), ts),
                VideoRandomApply(GaussianBlur([0.1, 2.0], crop_size), ts, p=0.5),
                normalize_video,
            ])

    def __call__(self, im_q, im_k, aux_info):
        im_q, aux_info = self.forward_flip(im_q, aux_info, suffix='_q')
        im_q = self.single_cal(im_q)

        im_k, aux_info = self.forward_flip(im_k, aux_info, suffix='_k')
        im_k = self.single_cal(im_k)

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
            for k in list(aux_info.keys()):
                if k.endswith(full_suffix):
                    sub_key_img = k.replace(self.flow_suffix, "flow_imgs")
                    sub_key_map = k.replace(self.flow_suffix, "motion_maps")
                    aux_info[sub_key_img] = self.visualizer(aux_info[k])
                    aux_info[sub_key_map] = self.motion_calculator(aux_info[k])

                    aux_info[k][to_apply] = torch.flip(aux_info[k][to_apply], [-1])
                    aux_info[sub_key_map][to_apply] = torch.flip(aux_info[sub_key_map][to_apply], [-1])
                    aux_info[sub_key_img][to_apply] = torch.flip(aux_info[sub_key_img][to_apply], [-1])
        if 'gt_bboxes'+suffix in aux_info:
            box_key = 'gt_bboxes'+suffix
            for i, flag in enumerate(to_apply):
                if flag:
                    aux_info[box_key][i] = self.bbox_hflip(aux_info[box_key][i])
        return clips, aux_info

    def single_cal(self, clips):
        # from (B, C, T, H, W) to (B, T, C, H, W)
        clips = clips.permute(0, 2, 1, 3, 4).contiguous()
        clips_batch = clips.view(-1, clips.shape[2], clips.shape[3], clips.shape[4])
        aug_clips = self.moco_augment_v2(clips_batch)
        aug_clips = aug_clips.view(clips.shape)
        # from (B, T, C, H, W) to (B, C, T, H, W)
        aug_clips = aug_clips.permute(0, 2, 1, 3, 4).contiguous()
        return aug_clips