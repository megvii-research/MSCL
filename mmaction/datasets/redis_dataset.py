import copy
import functools
import torch
import warnings
from collections import OrderedDict
from mmcv.utils import print_log

from ..core import (mean_average_precision, mean_class_accuracy,
                    mmit_mean_average_precision, top_k_accuracy)
from .base import BaseDataset
from .builder import DATASETS
from .pipelines import Compose
from .redis_cache import ActivityRedisCachedPickle
from mmaction.core.evaluation.visualizer import ClfVisualizer


class DataTransfer:
    def __init__(self, extra_keys) -> None:
        self.extra_keys = extra_keys
        
    def __call__(self, anno):
        item = {}
        item["nori_id_seq"] = anno["nori_id_seq"]
        item["total_frames"] = len(anno["nori_id_seq"])

        item["label"] = anno["label"]
        if "label_str" in anno:
            item["label_str"] = anno["label_str"]

        if "gt_bboxes" in self.extra_keys:
            item["gt_bboxes"] = anno["bboxs"]
        if "nids_flow" in self.extra_keys:
            item["nids_flow"] = anno["enc_flows"]
        if "nids_flow_img" in self.extra_keys:
            item["nids_flow_img"] = anno["imflows"]
        if "img_key" in self.extra_keys:
            item["img_key"] = anno["video_name"]
        if "chosen_idx" in self.extra_keys:
            item["chosen_idx"] = anno["chosen_idx"]

        return item
    

@DATASETS.register_module()
class RedisRawframeDataset(BaseDataset):
    """Rawframe dataset for action recognition.

    The dataset loads raw frames and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    the directory to frames of a video, total frames of the video and
    the label of a video, which are split with a whitespace.
    Example of a annotation file:

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        data_prefix (str | None): Path to a directory where videos are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        filename_tmpl (str): Template for each filename.
            Default: 'img_{:05}.jpg'.
        with_offset (bool): Determines whether the offset information is in
            ann_file. Default: False.
        multi_class (bool): Determines whether it is a multi-class
            recognition dataset. Default: False.
        num_classes (int | None): Number of classes in the dataset.
            Default: None.
        modality (str): Modality of data. Support 'RGB', 'Flow'.
            Default: 'RGB'.
        sample_by_class (bool): Sampling by class, should be set `True` when
            performing inter-class data balancing. Only compatible with
            `multi_class == False`. Only applies for training. Default: False.
        power (float): We support sampling data with the probability
            proportional to the power of its label frequency (freq ^ power)
            when sampling data. `power == 1` indicates uniformly sampling all
            data; `power == 0` indicates uniformly sampling all classes.
            Default: 0.
        dynamic_length (bool): If the dataset length is dynamic (used by
            ClassSpecificDistributedSampler). Default: False.
    """

    def __init__(self,
                 pkl_path,
                 pipeline,
                 data_prefix=None,
                 test_mode=False,
                 filename_tmpl='img_{:05}.jpg',
                 with_offset=False,
                 multi_class=False,
                 num_classes=None,
                 start_index=0,
                 modality='RGB',
                 sample_by_class=False,
                 power=0.,
                 dynamic_length=False,
                 redis_url="",
                 redis_master_url="",
                 extra_keys=list(),
                 visual_cfg=None):
        self.filename_tmpl = filename_tmpl
        self.with_offset = with_offset
        ann_file = None
        super().__init__(
            ann_file,
            pipeline,
            data_prefix,
            test_mode,
            multi_class,
            num_classes,
            start_index,
            modality,
            sample_by_class=sample_by_class,
            power=power,
            dynamic_length=dynamic_length)
        self.pkl_path = pkl_path
        self.redis_url = redis_url
        self.redis_master_url = redis_master_url
        self.data_transfer = DataTransfer(extra_keys)
        if visual_cfg:
            self.visualizer = ClfVisualizer(**visual_cfg)
        print(self.redis_url, self.redis_master_url)

    def load_annotations(self):
        return None

    @property
    @functools.lru_cache()
    def metas(self):
        video_infos = ActivityRedisCachedPickle(
            self.pkl_path, ttl=30 * 7 * 60 * 60 * 24,
            redis_url=self.redis_url, redis_master_url=self.redis_master_url,
        )
        return video_infos

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.data_transfer(self.metas[idx]))
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        # prepare tensor in getitem
        if self.multi_class:
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.data_transfer(self.metas[idx]))
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        # prepare tensor in getitem
        if self.multi_class:
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        return self.pipeline(results)

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.metas)

    def evaluate(self,
                 results,
                 metrics='top_k_accuracy',
                 metric_options=dict(top_k_accuracy=dict(topk=(1, 5))),
                 logger=None,
                 **deprecated_kwargs):
        """Perform evaluation for common datasets.

        Args:
            results (list): Output results.
            metrics (str | sequence[str]): Metrics to be performed.
                Defaults: 'top_k_accuracy'.
            metric_options (dict): Dict for metric options. Options are
                ``topk`` for ``top_k_accuracy``.
                Default: ``dict(top_k_accuracy=dict(topk=(1, 5)))``.
            logger (logging.Logger | None): Logger for recording.
                Default: None.
            deprecated_kwargs (dict): Used for containing deprecated arguments.
                See 'https://github.com/open-mmlab/mmaction2/pull/286'.

        Returns:
            dict: Evaluation results dict.
        """
        # Protect ``metric_options`` since it uses mutable value as default
        metric_options = copy.deepcopy(metric_options)

        if deprecated_kwargs != {}:
            warnings.warn(
                'Option arguments for metrics has been changed to '
                "`metric_options`, See 'https://github.com/open-mmlab/mmaction2/pull/286' "  # noqa: E501
                'for more details')
            metric_options['top_k_accuracy'] = dict(
                metric_options['top_k_accuracy'], **deprecated_kwargs)

        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            f'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = [
            'top_k_accuracy', 'mean_class_accuracy', 'mean_average_precision',
            'mmit_mean_average_precision', 'vis_mean_class_accuracy'
        ]

        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        eval_results = OrderedDict()
        gt_labels = []
        for anno in self.metas:
            label = self.data_transfer(anno)["label"]
            gt_labels.append(label)

        for metric in metrics:
            msg = f'Evaluating {metric} ...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'top_k_accuracy':
                topk = metric_options.setdefault('top_k_accuracy',
                                                 {}).setdefault(
                                                     'topk', (1, 5))
                if not isinstance(topk, (int, tuple)):
                    raise TypeError('topk must be int or tuple of int, '
                                    f'but got {type(topk)}')
                if isinstance(topk, int):
                    topk = (topk, )

                top_k_acc = top_k_accuracy(results, gt_labels, topk)
                log_msg = []
                for k, acc in zip(topk, top_k_acc):
                    eval_results[f'top{k}_acc'] = acc
                    log_msg.append(f'\ntop{k}_acc\t{acc:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric in ['mean_class_accuracy', 'vis_mean_class_accuracy']:
                if metric == 'mean_class_accuracy':
                    mean_acc = mean_class_accuracy(results, gt_labels)
                elif metric == 'vis_mean_class_accuracy':
                    mean_acc = self.visualizer.visualize(results, gt_labels)
                else:
                    raise ValueError(f"Not support metric {metric} now!")
                eval_results['mean_class_accuracy'] = mean_acc
                log_msg = f'\nmean_acc\t{mean_acc:.4f}'
                print_log(log_msg, logger=logger)
                continue

            if metric in [
                    'mean_average_precision', 'mmit_mean_average_precision'
            ]:
                gt_labels = [
                    self.label2array(self.num_classes, label)
                    for label in gt_labels
                ]
                if metric == 'mean_average_precision':
                    mAP = mean_average_precision(results, gt_labels)
                    eval_results['mean_average_precision'] = mAP
                    log_msg = f'\nmean_average_precision\t{mAP:.4f}'
                elif metric == 'mmit_mean_average_precision':
                    mAP = mmit_mean_average_precision(results, gt_labels)
                    eval_results['mmit_mean_average_precision'] = mAP
                    log_msg = f'\nmmit_mean_average_precision\t{mAP:.4f}'
                print_log(log_msg, logger=logger)
                continue

        return eval_results


@DATASETS.register_module()
class RedisRawframe2BranchDataset(RedisRawframeDataset):
    """Rawframe dataset for action recognition.

    The dataset loads raw frames and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    the directory to frames of a video, total frames of the video and
    the label of a video, which are split with a whitespace.
    Example of a annotation file:

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        data_prefix (str | None): Path to a directory where videos are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        filename_tmpl (str): Template for each filename.
            Default: 'img_{:05}.jpg'.
        with_offset (bool): Determines whether the offset information is in
            ann_file. Default: False.
        multi_class (bool): Determines whether it is a multi-class
            recognition dataset. Default: False.
        num_classes (int | None): Number of classes in the dataset.
            Default: None.
        modality (str): Modality of data. Support 'RGB', 'Flow'.
            Default: 'RGB'.
        sample_by_class (bool): Sampling by class, should be set `True` when
            performing inter-class data balancing. Only compatible with
            `multi_class == False`. Only applies for training. Default: False.
        power (float): We support sampling data with the probability
            proportional to the power of its label frequency (freq ^ power)
            when sampling data. `power == 1` indicates uniformly sampling all
            data; `power == 0` indicates uniformly sampling all classes.
            Default: 0.
        dynamic_length (bool): If the dataset length is dynamic (used by
            ClassSpecificDistributedSampler). Default: False.
    """

    def __init__(self,
                 pkl_path,
                 pipeline1,
                 pipeline2,
                 data_prefix=None,
                 test_mode=False,
                 filename_tmpl='img_{:05}.jpg',
                 with_offset=False,
                 multi_class=False,
                 num_classes=None,
                 start_index=0,
                 modality='RGB',
                 sample_by_class=False,
                 power=0.,
                 dynamic_length=False,
                 redis_url="",
                 redis_master_url=""):
        super().__init__(
            pkl_path,
            pipeline1,
            data_prefix=data_prefix,
            test_mode=test_mode,
            filename_tmpl=filename_tmpl,
            with_offset=with_offset,
            multi_class=multi_class,
            num_classes=num_classes,
            start_index=start_index,
            modality=modality,
            sample_by_class=sample_by_class,
            power=power,
            dynamic_length=dynamic_length,
            redis_url=redis_url,
            redis_master_url=redis_master_url,
            )

        self.pipeline2 = Compose(pipeline2)

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.data_transfer(self.metas[idx]))
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        # prepare tensor in getitem
        if self.multi_class:
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        data1 = self.pipeline(results)
        data2 = self.pipeline2(results)
        data = {}
        for k in data1.keys():
            data[k] = torch.stack((data1[k], data2[k]), 0)
        return data

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.data_transfer(self.metas[idx]))
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        # prepare tensor in getitem
        if self.multi_class:
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        data1 = self.pipeline(results)
        data2 = self.pipeline2(results)
        data = {}
        for k in data1.keys():
            data[k] = torch.stack((data1[k], data2[k]), 0)
        return data
