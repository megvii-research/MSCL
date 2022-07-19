# Copyright (c) OpenMMLab. All rights reserved.
from concurrent.futures import thread
import random

import mmcv
import torch
import numpy as np
import torch.nn.functional as F
from scipy import ndimage, misc

from collections.abc import Sequence
from ..builder import PIPELINES
from .augmentations import RandomCrop, _init_lazy_if_proper

def cal_motion_map(flow, sl=14, ds_rate=2):
    # flow: ndarray with h, w, 2
    h, w = flow.shape[:2]
    u, v = flow[..., 0], flow[..., 1]

    # sobel_u_x = cv2.Sobel(u, -1, 1, 0, ksize=3)   # need to be h,w,c
    sobel_u_x = ndimage.sobel(u, axis=-1)   # h,w is ok
    sobel_u_y = ndimage.sobel(u, axis=0)
    sobel_v_x = ndimage.sobel(v, axis=-1)
    sobel_v_y = ndimage.sobel(v, axis=0)
    motion_map = np.sqrt(np.square(sobel_u_x)+np.square(sobel_u_y)+np.square(sobel_v_x)+np.square(sobel_v_y))   # h,w
    motion_map = torch.from_numpy(motion_map).unsqueeze(0).unsqueeze(0).float()
    motion_map = F.avg_pool2d(motion_map, kernel_size=(sl, sl), stride=(sl, sl))
    motion_map = F.interpolate(motion_map, size=(h//ds_rate, w//ds_rate), mode='bilinear')
    
    return motion_map.squeeze(0).squeeze(0)


@PIPELINES.register_module()
class MCLRandomResizedCrop(RandomCrop):
    def __init__(self,
                 area_range=(0.08, 1.0),
                 aspect_ratio_range=(3 / 4, 4 / 3),
                 lazy=False, flow_key=None, th_rate=0.8, fast=True, sl=14):
        self.area_range = area_range
        self.aspect_ratio_range = aspect_ratio_range
        self.lazy = lazy
        self.flow_key = flow_key
        self.fast = fast
        self.sl = sl
        self.th_rate = th_rate

        if not mmcv.is_tuple_of(self.area_range, float):
            raise TypeError(f'Area_range must be a tuple of float, '
                            f'but got {type(area_range)}')
        if not mmcv.is_tuple_of(self.aspect_ratio_range, float):
            raise TypeError(f'Aspect_ratio_range must be a tuple of float, '
                            f'but got {type(aspect_ratio_range)}')

    def get_crop_bbox(self, motion_map,
                      area_range,
                      aspect_ratio_range,
                      max_attempts=20):
        assert 0 < area_range[0] <= area_range[1] <= 1
        assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]

        img_h, img_w = motion_map.shape
        area = img_h * img_w
        v_topk = motion_map.view(-1).topk(k=int(area*self.th_rate))[0][-1]

        min_ar, max_ar = aspect_ratio_range
        aspect_ratios = np.exp(
            np.random.uniform(
                np.log(min_ar), np.log(max_ar), size=max_attempts))
        target_areas = np.random.uniform(*area_range, size=max_attempts) * area
        candidate_crop_w = np.round(np.sqrt(target_areas *
                                            aspect_ratios)).astype(np.int32)
        candidate_crop_h = np.round(np.sqrt(target_areas /
                                            aspect_ratios)).astype(np.int32)

        check_th = 0.2
        cur_rate, mul, rate_min = 1, 0.92, 0.5
        for i in range(max_attempts):
            crop_w = candidate_crop_w[i]
            crop_h = candidate_crop_h[i]
            if crop_h <= img_h and crop_w <= img_w:
                x_offset = random.randint(0, img_w - crop_w)
                y_offset = random.randint(0, img_h - crop_h)
                cur_map = motion_map[y_offset:y_offset+crop_h, x_offset:x_offset+crop_w].reshape(-1)
                if cur_map.topk(k=int(cur_map.shape[0]*check_th) )[0][-1] > v_topk*cur_rate:
                    return x_offset, y_offset, x_offset + crop_w, y_offset + crop_h
                else:
                    cur_rate = max(cur_rate*mul, rate_min)
                # print(i, motion_map[y_offset:y_offset+crop_h, x_offset:x_offset+crop_w].sum(), cur_rate, v_topk, motion_map[y_offset:y_offset+crop_h, x_offset:x_offset+crop_w].max(), \
                #     motion_map[y_offset:y_offset+crop_h, x_offset:x_offset+crop_w].min(), cur_rate*crop_w*crop_h*v_topk)
                # if motion_map[y_offset:y_offset+crop_h, x_offset:x_offset+crop_w].sum() > cur_rate*crop_w*crop_h*v_topk:
                #     return x_offset, y_offset, x_offset + crop_w, y_offset + crop_h
                # else:
                #     cur_rate *= mul

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

    def single_cal(self, imgs, results, flows, suffix='_q'):
        img_h, img_w = results['img_shape']
        if self.fast:
            num_flow = len(flows)
            fid = np.random.randint(num_flow)
            cur_flow = flows[fid]
        else:
            raise NotImplementedError("Not support now!!")
        motion_map = cal_motion_map(cur_flow, sl=self.sl, ds_rate=4)

        left, top, right, bottom = self.get_crop_bbox(
            motion_map, self.area_range, self.aspect_ratio_range)
        mo_h, mo_w = motion_map.shape
        h_rate, w_rate = img_h/mo_h, img_w/mo_w 
        left, top, right, bottom = \
            int(round(left*w_rate)), int(round(top*h_rate)), int(round(right*w_rate)), int(round(bottom*h_rate))

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

        if 'flow_im_rate' in results:
            h_rate, w_rate = results['flow_im_rate']
        else:
            flow_h, flow_w = flows[0].shape[0], flows[0].shape[1]
            h_rate, w_rate = flow_h/img_h, flow_w/img_w 
        left_f, top_f, right_f, bottom_f = \
            int(round(left*w_rate)), int(round(top*h_rate)), int(round(right*w_rate)), int(round(bottom*h_rate))
        crop_bbox_flow = np.array([left_f, top_f, right_f, bottom_f])
        results[self.flow_key + suffix] = self._crop_imgs(flows, crop_bbox_flow)

        if 'gt_bboxes' in results:
            assert not self.lazy
            results = self._all_box_crop(results, results['crop_bbox' + suffix], suffix=suffix)

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
        # Merge
        # results['imgs'] = [results['imgs_q'], results['imgs_k']]
        # if self.flow_key in results:
        #     results[self.flow_key] = [results[self.flow_key + '_q'], results[self.flow_key + '_k']]
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