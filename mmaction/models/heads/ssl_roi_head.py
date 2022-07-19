# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmaction.core.bbox import bbox2result
from mmaction.utils import import_module_error_class
from ..builder import HEADS

try:
    from mmdet.core.bbox import bbox2roi
    from mmdet.models.roi_heads import StandardRoIHead
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False

if mmdet_imported:

    @HEADS.register_module()
    class SSLRoIHead(StandardRoIHead):
        """
        call forward_train (e.g. self.rpn_head.forward_train in FasterRCNN)
        -> assign + sample
        -> _bbox_forward_train(defined here)
            -> bbox2roi(bid + boxes) -> _bbox_forward(ft->pred) -> bbox_head(target + loss)
        """

        # def _bbox_forward(self, x, rois, img_metas):
        #     """Defines the computation performed to get bbox predictions.

        #     Args:
        #         x (torch.Tensor): The input tensor.
        #         rois (torch.Tensor): The regions of interest.
        #         img_metas (list): The meta info of images

        #     Returns:
        #         dict: bbox predictions with features and classification scores.
        #     """
        #     bbox_feat, global_feat = self.bbox_roi_extractor(x, rois)

        #     if self.with_shared_head:
        #         bbox_feat = self.shared_head(
        #             bbox_feat,
        #             feat=global_feat,
        #             rois=rois,
        #             img_metas=img_metas)

        #     cls_score, bbox_pred = self.bbox_head(bbox_feat)

        #     bbox_results = dict(
        #         cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feat)
        #     return bbox_results

        # def _bbox_forward_train(self, x, sampling_results, gt_bboxes,
        #                         gt_labels, img_metas):
        #     """Run forward function and calculate loss for box head in
        #     training."""
        #     rois = bbox2roi([res.bboxes for res in sampling_results])   # +batch id | res.bboxes -> list(Tensor)
        #     bbox_results = self._bbox_forward(x, rois, img_metas)

        #     bbox_targets = self.bbox_head.get_targets(sampling_results,
        #                                               gt_bboxes, gt_labels,
        #                                               self.train_cfg)
        #     loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
        #                                     bbox_results['bbox_pred'], rois,
        #                                     *bbox_targets)

        #     bbox_results.update(loss_bbox=loss_bbox)
        #     return bbox_results

        def _bbox_extract_feature(self, x, img_metas, level=-2, suffix='_q'):
            # x -> mlvl feature
            x = x[level]
            # For similarity, remove assigner and sampler
            # rois = bbox2roi([res.bboxes for res in sampling_results])
            # Please set the train_cfg of SSLRoIHead to None to avoid
            # init_assigner
            sampling_results = img_metas["gt_bboxes" + suffix]
            # Assigner -> only support one bbox
            for idx in range(len(sampling_results)):
                if sampling_results[idx].numel():
                    # TDOO: support multiple frames
                    sampling_results[idx] = sampling_results[idx][:1]
                else:
                    sampling_results[idx] = x.new_zeros((1, 4))
            # ---------------------------------
            rois = bbox2roi([res for res in sampling_results])
            bbox_feat, global_feat = self.bbox_roi_extractor(x, rois)
            if self.with_shared_head:
                bbox_feat = self.shared_head(
                    bbox_feat,
                    feat=global_feat,
                    rois=rois,
                    img_metas=img_metas)
            return bbox_feat, {'rois'+suffix:rois}
else:
    # Just define an empty class, so that __init__ can import it.
    @import_module_error_class('mmdet')
    class SSLRoIHead:
        pass
