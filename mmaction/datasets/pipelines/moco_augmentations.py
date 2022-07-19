# Copyright (c) OpenMMLab. All rights reserved.
import random

import mmcv
import numpy as np

from collections.abc import Sequence
from ..builder import PIPELINES
from .augmentations import RandomCrop, _init_lazy_if_proper

@PIPELINES.register_module()
class MoCoRandomResizedCrop(RandomCrop):
    """Random crop that specifics the area and height-weight ratio range.

    Required keys in results are "img_shape", "crop_bbox", "imgs" (optional),
    "keypoint" (optional), added or modified keys are "imgs", "keypoint",
    "crop_bbox" and "lazy"; Required keys in "lazy" are "flip", "crop_bbox",
    added or modified key is "crop_bbox".

    Args:
        area_range (Tuple[float]): The candidate area scales range of
            output cropped images. Default: (0.08, 1.0).
        aspect_ratio_range (Tuple[float]): The candidate aspect ratio range of
            output cropped images. Default: (3 / 4, 4 / 3).
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self,
                 area_range=(0.08, 1.0),
                 aspect_ratio_range=(3 / 4, 4 / 3),
                 lazy=False, flow_key=None):
        self.area_range = area_range
        self.aspect_ratio_range = aspect_ratio_range
        self.lazy = lazy
        self.flow_key = flow_key

        if not mmcv.is_tuple_of(self.area_range, float):
            raise TypeError(f'Area_range must be a tuple of float, '
                            f'but got {type(area_range)}')
        if not mmcv.is_tuple_of(self.aspect_ratio_range, float):
            raise TypeError(f'Aspect_ratio_range must be a tuple of float, '
                            f'but got {type(aspect_ratio_range)}')

    @staticmethod
    def get_crop_bbox(img_shape,
                      area_range,
                      aspect_ratio_range,
                      max_attempts=10):
        """Get a crop bbox given the area range and aspect ratio range.

        Args:
            img_shape (Tuple[int]): Image shape
            area_range (Tuple[float]): The candidate area scales range of
                output cropped images. Default: (0.08, 1.0).
            aspect_ratio_range (Tuple[float]): The candidate aspect
                ratio range of output cropped images. Default: (3 / 4, 4 / 3).
                max_attempts (int): The maximum of attempts. Default: 10.
            max_attempts (int): Max attempts times to generate random candidate
                bounding box. If it doesn't qualified one, the center bounding
                box will be used.
        Returns:
            (list[int]) A random crop bbox within the area range and aspect
            ratio range.
        """
        assert 0 < area_range[0] <= area_range[1] <= 1
        assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]

        img_h, img_w = img_shape
        area = img_h * img_w

        min_ar, max_ar = aspect_ratio_range
        aspect_ratios = np.exp(
            np.random.uniform(
                np.log(min_ar), np.log(max_ar), size=max_attempts))
        target_areas = np.random.uniform(*area_range, size=max_attempts) * area
        candidate_crop_w = np.round(np.sqrt(target_areas *
                                            aspect_ratios)).astype(np.int32)
        candidate_crop_h = np.round(np.sqrt(target_areas /
                                            aspect_ratios)).astype(np.int32)

        for i in range(max_attempts):
            crop_w = candidate_crop_w[i]
            crop_h = candidate_crop_h[i]
            if crop_h <= img_h and crop_w <= img_w:
                x_offset = random.randint(0, img_w - crop_w)
                y_offset = random.randint(0, img_h - crop_h)
                return x_offset, y_offset, x_offset + crop_w, y_offset + crop_h

        # Fallback
        crop_size = min(img_h, img_w)
        x_offset = (img_w - crop_size) // 2
        y_offset = (img_h - crop_size) // 2
        return x_offset, y_offset, x_offset + crop_size, y_offset + crop_size

    def _all_box_crop(self, results, crop_bbox, suffix=''):
        results['gt_bboxes' + suffix] = self._box_crop(results['gt_bboxes'], crop_bbox)
        if 'proposals' in results and results['proposals'] is not None:
            assert results['proposals'].shape[1] == 4
            results['proposals' + suffix] = self._box_crop(results['proposals'],
                                                  crop_bbox)
        return results

    def _del_box_keys(self, results):
        del results['gt_bboxes']
        if 'proposals' in results:
            del results['proposals']

    def single_cal(self, imgs, results, flows=None, suffix='_q'):
        img_h, img_w = results['img_shape']

        left, top, right, bottom = self.get_crop_bbox(
            (img_h, img_w), self.area_range, self.aspect_ratio_range)
        new_h, new_w = bottom - top, right - left

        if 'crop_quadruple' not in results:
            results['crop_quadruple' + suffix] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)
        else:
            results['crop_quadruple' + suffix] = results['crop_quadruple']

        x_ratio, y_ratio = left / img_w, top / img_h
        w_ratio, h_ratio = new_w / img_w, new_h / img_h

        old_crop_quadruple = results['crop_quadruple' + suffix]
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_x_ratio
        ]
        results['crop_quadruple' + suffix] = np.array(
            new_crop_quadruple, dtype=np.float32)

        crop_bbox = np.array([left, top, right, bottom])
        results['crop_bbox' + suffix] = crop_bbox
        results['img_shape' + suffix] = (new_h, new_w)

        if not self.lazy:
            if 'keypoint' in results:
                raise NotImplementedError("Not support keypoint now!")
            if 'imgs' in results:
                results['imgs' + suffix] = self._crop_imgs(imgs, crop_bbox)
        else:
            raise NotImplementedError("Not support lazy op now!")

        if 'gt_bboxes' in results:
            assert not self.lazy
            results = self._all_box_crop(results, results['crop_bbox' + suffix], suffix=suffix)

        if flows is not None:
            if 'flow_im_rate' in results:
                h_rate, w_rate = results['flow_im_rate']
            else:
                flow_h, flow_w = flows[0].shape[0], flows[0].shape[1]
                h_rate, w_rate = flow_h/img_h, flow_w/img_w 
            left_f, top_f, right_f, bottom_f = \
                int(round(left*w_rate)), int(round(top*h_rate)), int(round(right*w_rate)), int(round(bottom*h_rate))
            crop_bbox_flow = np.array([left_f, top_f, right_f, bottom_f])
            results[self.flow_key + suffix] = self._crop_imgs(flows, crop_bbox_flow)

        return results


    def __call__(self, results):
        """Performs the RandomResizeCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)
        if 'keypoint' in results:
            assert not self.lazy, ('Keypoint Augmentations are not compatible '
                                   'with lazy == True')

        clip_len = results['clip_len']
        imgs = results['imgs']
        imgs_len = len(imgs)
        if clip_len == imgs_len:
            imgs_q, imgs_k = imgs, imgs
        else:
            imgs_q, imgs_k = imgs[:imgs_len//2], imgs[imgs_len//2:]
        flows_q, flows_k = None, None
        if self.flow_key:
            flows = results[self.flow_key]
            if clip_len == imgs_len:
                flows_q, flows_k = flows, flows
            else:
                flows_len = len(flows)
                flows_q, flows_k = flows[:flows_len//2], flows[flows_len//2:]
        results = self.single_cal(imgs_q, results, flows_q, suffix='_q')
        results = self.single_cal(imgs_k, results, flows_k, suffix='_k')

        results['img_shape'] = results['img_shape_q']
        del results['imgs']
        if self.flow_key:
            del results[self.flow_key]
        if 'gt_bboxes' in results:
            self._del_box_keys(results)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'area_range={self.area_range}, '
                    f'aspect_ratio_range={self.aspect_ratio_range}, '
                    f'lazy={self.lazy})')
        return repr_str


@PIPELINES.register_module()
class MoCoResize:
    """Resize images to a specific size.

    Required keys are "img_shape", "modality", "imgs" (optional), "keypoint"
    (optional), added or modified keys are "imgs", "img_shape", "keep_ratio",
    "scale_factor", "lazy", "resize_size". Required keys in "lazy" is None,
    added or modified key is "interpolation".

    Args:
        scale (float | Tuple[int]): If keep_ratio is True, it serves as scaling
            factor or maximum size:
            If it is a float number, the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, the image will
            be rescaled as large as possible within the scale.
            Otherwise, it serves as (w, h) of output size.
        keep_ratio (bool): If set to True, Images will be resized without
            changing the aspect ratio. Otherwise, it will resize images to a
            given size. Default: True.
        interpolation (str): Algorithm used for interpolation:
            "nearest" | "bilinear". Default: "bilinear".
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self,
                 scale,
                 keep_ratio=False,
                 interpolation='bilinear',
                 lazy=False,
                 suffix='', flow_key='flow_imgs'):
        if isinstance(scale, float):
            if scale <= 0:
                raise ValueError(f'Invalid scale {scale}, must be positive.')
        elif isinstance(scale, tuple):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            if max_short_edge == -1:
                # assign np.inf to long edge for rescaling short edge later.
                scale = (np.inf, max_long_edge)
        else:
            raise TypeError(
                f'Scale must be float or tuple of int, but got {type(scale)}')
        self.scale = scale
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation
        self.lazy = lazy
        self.suffix = suffix
        self.flow_key = flow_key

    def _resize_imgs(self, imgs, new_w, new_h):
        return [
            mmcv.imresize(
                img, (new_w, new_h), interpolation=self.interpolation)
            for img in imgs
        ]

    @staticmethod
    def _resize_kps(kps, scale_factor):
        return kps * scale_factor

    @staticmethod
    def _box_resize(box, scale_factor):
        """Rescale the bounding boxes according to the scale_factor.

        Args:
            box (np.ndarray): The bounding boxes.
            scale_factor (np.ndarray): The scale factor used for rescaling.
        """
        assert len(scale_factor) == 2
        scale_factor = np.concatenate([scale_factor, scale_factor])
        return box * scale_factor

    def _fn(self, name):
        return name + self.suffix

    def __call__(self, results):
        """Performs the Resize augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        _init_lazy_if_proper(results, self.lazy)
        if self._fn('keypoint') in results:
            assert not self.lazy, ('Keypoint Augmentations are not compatible '
                                   'with lazy == True')

        if self._fn('scale_factor') not in results:
            results[self._fn('scale_factor')] = np.array([1, 1], dtype=np.float32)
        img_h, img_w = results[self._fn('img_shape')]

        if self.keep_ratio:
            new_w, new_h = mmcv.rescale_size((img_w, img_h), self.scale)
        else:
            new_w, new_h = self.scale

        self.scale_factor = np.array([new_w / img_w, new_h / img_h],
                                     dtype=np.float32)

        results[self._fn('img_shape')] = (new_h, new_w)
        results[self._fn('keep_ratio')] = self.keep_ratio
        results[self._fn('scale_factor')] = results[self._fn('scale_factor')] * self.scale_factor

        if not self.lazy:
            if self._fn('imgs') in results:
                results[self._fn('imgs')] = self._resize_imgs(results[self._fn('imgs')], new_w,
                                                    new_h)
            if self._fn('keypoint') in results:
                results[self._fn('keypoint')] = self._resize_kps(results[self._fn('keypoint')],
                                                       self.scale_factor)
            if self.flow_key:
                results[self._fn(self.flow_key)] = self._resize_imgs(results[self._fn(self.flow_key)], new_w,
                                                    new_h)
        else:
            raise NotImplementedError("Not support now!")

        if self._fn('gt_bboxes') in results:
            assert not self.lazy
            results[self._fn('gt_bboxes')] = self._box_resize(results[self._fn('gt_bboxes')],
                                                    self.scale_factor)
            if self._fn('proposals') in results and results[self._fn('proposals')] is not None:
                assert results[self._fn('proposals')].shape[1] == 4
                results[self._fn('proposals')] = self._box_resize(
                    results[self._fn('proposals')], self.scale_factor)

        # TODO: make following code alone 
        if self.suffix == '_k':
            results['imgs'] = [results['imgs_q'], results['imgs_k']]
            if self.flow_key:
                results[self.flow_key] = [results[self.flow_key + '_q'], results[self.flow_key + '_k']]
            results["img_shape"] = results["img_shape_q"]

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'scale={self.scale}, keep_ratio={self.keep_ratio}, '
                    f'interpolation={self.interpolation}, '
                    f'lazy={self.lazy})')
        return repr_str


@PIPELINES.register_module()
class MoCoNormalize:
    """Div 255 + transpose to CTHW for imgs and flow_imgs.
    """

    def __init__(self, ori_flow=False):
        self.ori_flow = ori_flow

    def __call__(self, results):
        n, nf = len(results['imgs'][0]), len(results['flow_imgs'][0])
        h, w, c = results['imgs'][0][0].shape
        hf, wf, cf = results['flow_imgs'][0][0].shape
        assert hf == h and wf == w

        for idx in range(2):
            imgs = np.empty((n, h, w, c), dtype=np.float32)
            for i, img in enumerate(results['imgs'][idx]):
                imgs[i] = img
                imgs[i] /= 255.0
            results['imgs'][idx] = imgs.transpose((3, 0, 1, 2))
            if 'flow_imgs' in results:
                flows = np.empty((nf, hf, wf, cf), dtype=np.float32)
                for i, flow in enumerate(results['flow_imgs'][idx]):
                    flows[i] = flow
                    if not self.ori_flow:
                        flows[i] /= 255.0
                results['flow_imgs'][idx] = flows.transpose((3, 0, 1, 2))
        return results


@PIPELINES.register_module()
class MoCoNormalizeV2:
    """Div 255 + transpose to CTHW for imgs and flow_imgs +
    transpose to CTHW for flows.
    """

    def __init__(self, ori_flow=False):
        self.ori_flow = ori_flow

    def __call__(self, results):
        n = len(results['imgs'][0])
        h, w, c = results['imgs'][0][0].shape

        for idx in range(2):
            imgs = np.empty((n, h, w, c), dtype=np.float32)
            for i, img in enumerate(results['imgs'][idx]):
                imgs[i] = img
                imgs[i] /= 255.0
            results['imgs'][idx] = imgs.transpose((3, 0, 1, 2))
            if 'flow_imgs' in results:
                nf = len(results['flow_imgs'][0])
                hf, wf, cf = results['flow_imgs'][idx][0].shape
                assert hf == h and wf == w
                flow_imgs = np.empty((nf, hf, wf, cf), dtype=np.float32)
                for i, flow_img in enumerate(results['flow_imgs'][idx]):
                    flow_imgs[i] = flow_img
                    if not self.ori_flow:
                        flow_imgs[i] /= 255.0
                results['flow_imgs'][idx] = flow_imgs.transpose((3, 0, 1, 2))
            if 'flows' in results:
                nf = len(results['flows'][idx])
                hf, wf, cf = results['flows'][idx][0].shape
                assert hf == h and wf == w
                flows = np.empty((nf, hf, wf, cf), dtype=np.float32)
                for i, flow in enumerate(results['flows'][idx]):
                    flows[i] = flow
                results['flows'][idx] = flows.transpose((3, 0, 1, 2))
        return results
