# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import mmcv
import pickle
import numpy as np
import torch

from functools import partial
from torch.utils.data import Sampler, DataLoader

from mmcv import Config, DictAction
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel

from mmaction.datasets.pipelines import Compose
from mmaction.datasets import build_dataset
from mmaction.utils import GradCAM
from mmaction.models import build_recognizer

def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 GradCAM demo')

    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file/url')
    # parser.add_argument('video', help='video file/url or rawframes directory')
    parser.add_argument(
        '--use-frames',
        default=False,
        action='store_true',
        help='whether to use rawframes as input')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--target-layer-name',
        type=str,
        default='backbone/layer4/1/relu',
        help='GradCAM target layer name')
    parser.add_argument('--out-filename', default=None, help='output filename')
    parser.add_argument('--metapath', default=None, help='path of meta indices')
    parser.add_argument('--fps', default=5, type=int)
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--target-resolution',
        nargs=2,
        default=None,
        type=int,
        help='Target resolution (w, h) for resizing the frames when using a '
        'video as input. If either dimension is set to -1, the frames are '
        'resized by keeping the existing aspect ratio')
    parser.add_argument(
        '--resize-algorithm',
        default='bilinear',
        help='resize algorithm applied to generate video & gif')

    args = parser.parse_args()
    return args

def init_recognizer(config,
                    checkpoint=None,
                    device='cuda:0',
                    use_frames=False):
    """Initialize a recognizer from config file.

    Args:
        config (str | :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str | None, optional): Checkpoint path/url. If set to None,
            the model will not load any weights. Default: None.
        device (str | :obj:`torch.device`): The desired device of returned
            tensor. Default: 'cuda:0'.
        use_frames (bool): Whether to use rawframes as input. Default:False.

    Returns:
        nn.Module: The constructed recognizer.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if ((use_frames and config.dataset_type == 'VideoDataset')
            or (not use_frames and config.dataset_type != 'VideoDataset')):
        input_type = 'rawframes' if use_frames else 'video'
        raise RuntimeError('input data type should be consist with the '
                           f'dataset type in config, but got input type '
                           f"'{input_type}' and dataset type "
                           f"'{config.dataset_type}'")

    # pretrained model is unnecessary since we directly load checkpoint later
    config.model.backbone.pretrained = None
    model = build_recognizer(config.model, test_cfg=config.get('test_cfg'))
    print(type(model))

    if checkpoint is not None:
        load_checkpoint(model, checkpoint, map_location=device)
    model.cfg = config
    model.to(device)
    model.eval()
    return model

def build_inputs(model, video_path, use_frames=False):
    """build inputs for GradCAM.

    Note that, building inputs for GradCAM is exactly the same as building
    inputs for Recognizer test stage. Codes from `inference_recognizer`.

    Args:
        model (nn.Module): Recognizer model.
        video_path (str): video file/url or rawframes directory.
        use_frames (bool): whether to use rawframes as input.
    Returns:
        dict: Both GradCAM inputs and Recognizer test stage inputs,
            including two keys, ``imgs`` and ``label``.
    """
    if not (osp.exists(video_path) or video_path.startswith('http')):
        raise RuntimeError(f"'{video_path}' is missing")

    if osp.isfile(video_path) and use_frames:
        raise RuntimeError(
            f"'{video_path}' is a video file, not a rawframe directory")
    if osp.isdir(video_path) and not use_frames:
        raise RuntimeError(
            f"'{video_path}' is a rawframe directory, not a video file")

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    # build the data pipeline
    test_pipeline = cfg.data.test.pipeline
    test_pipeline = Compose(test_pipeline)
    # prepare data
    if use_frames:
        filename_tmpl = cfg.data.test.get('filename_tmpl', 'img_{:05}.jpg')
        modality = cfg.data.test.get('modality', 'RGB')
        start_index = cfg.data.test.get('start_index', 1)
        data = dict(
            frame_dir=video_path,
            total_frames=len(os.listdir(video_path)),
            label=-1,
            start_index=start_index,
            filename_tmpl=filename_tmpl,
            modality=modality)
    else:
        start_index = cfg.data.test.get('start_index', 0)
        data = dict(
            filename=video_path,
            label=-1,
            start_index=start_index,
            modality='RGB')
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    return data


def _resize_frames(frame_list,
                   scale,
                   keep_ratio=True,
                   interpolation='bilinear'):
    """resize frames according to given scale.

    Codes are modified from `mmaction2/datasets/pipelines/augmentation.py`,
    `Resize` class.

    Args:
        frame_list (list[np.ndarray]): frames to be resized.
        scale (tuple[int]): If keep_ratio is True, it serves as scaling
            factor or maximum size: the image will be rescaled as large
            as possible within the scale. Otherwise, it serves as (w, h)
            of output size.
        keep_ratio (bool): If set to True, Images will be resized without
            changing the aspect ratio. Otherwise, it will resize images to a
            given size. Default: True.
        interpolation (str): Algorithm used for interpolation:
            "nearest" | "bilinear". Default: "bilinear".
    Returns:
        list[np.ndarray]: Both GradCAM and Recognizer test stage inputs,
            including two keys, ``imgs`` and ``label``.
    """
    if scale is None or (scale[0] == -1 and scale[1] == -1):
        return frame_list
    scale = tuple(scale)
    max_long_edge = max(scale)
    max_short_edge = min(scale)
    if max_short_edge == -1:
        scale = (np.inf, max_long_edge)

    img_h, img_w, _ = frame_list[0].shape

    if keep_ratio:
        new_w, new_h = mmcv.rescale_size((img_w, img_h), scale)
    else:
        new_w, new_h = scale

    frame_list = [
        mmcv.imresize(img, (new_w, new_h), interpolation=interpolation)
        for img in frame_list
    ]

    return frame_list


class IdxSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """
    def __init__(self, indices) -> None:
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def get_indices(dataset, c_type, const=10, max_num=10, tar_classes=None, file_path=None):
    if c_type == 'const':
        if isinstance(const, list):
            return dict(const=const)
        else:
            return dict(test=[const])
    elif c_type == 'class':
        res, count = [], {c:0 for c in tar_classes}
        for i, meta in enumerate(dataset.metas):
            if meta['label'] in count and count[meta['label']] < max_num:
                res.append(i)
                count[meta['label']] += 1
        return dict(c=res)
    elif c_type == 'file':
        with open(file_path, "rb") as f:
            res = pickle.load(f)
        return res


def main():
    args = parse_args()

    # assign the desired device.
    device = torch.device(args.device)

    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.cfg_options)

    # build the recognizer from a config file and checkpoint file/url
    model = init_recognizer(
        cfg, args.checkpoint, device=device, use_frames=args.use_frames)

    # build the dataset
    test_dataset = build_dataset(cfg.data.test, dict(test_mode=True))   # default_args

    # Sampler -- two types: class_list + max_number or real indices
    if args.metapath is not None:
        inds_name = os.path.splitext(os.path.basename(args.metapath))[0]
        indices_dict = get_indices(test_dataset, 'file', file_path=args.metapath)
    else:
        inds_name = "base_frames"
        const = [22885, 1044, 2020, 158, 15, 139, 157, 163, 306]
        indices_dict = get_indices(test_dataset, 'const', const=const, max_num=15)
        # inds_name = "base"
        # indices_dict = get_indices(test_dataset, 'class', tar_classes=[49,50,57,158], max_num=15)
    gradcam = GradCAM(model, args.target_layer_name)
    for k, indices in indices_dict.items():
        print(k, len(indices))
        sampler = IdxSampler(indices)
        test_loader = DataLoader(
                                test_dataset, batch_size=1, sampler=sampler, num_workers=0, 
                                collate_fn=partial(collate, samples_per_gpu=1), pin_memory=True, drop_last=False)
        assert len(indices) == len(test_loader), f"{len(indices)} vs {len(test_loader)}"
        cur_dirname = args.out_filename.replace('split', f"{inds_name}_{k}")
        os.makedirs(os.path.dirname(cur_dirname), exist_ok=True)
        for i, data in enumerate(test_loader):
            # data = collate([data], samples_per_gpu=1)
            if next(model.parameters()).is_cuda:
                # scatter to specified GPU
                data = scatter(data, [device])[0]           # for Not MMDP/MMDDP, data will not be conversed automatically.
            results = gradcam(data)
            assert results[0].shape[0] == 1, f"Find size with {results[0].shape}"
            if args.out_filename is not None:
                try:
                    from moviepy.editor import ImageSequenceClip
                except ImportError:
                    raise ImportError('Please install moviepy to enable output file.')

                # frames_batches shape [B, T, H, W, 3], in RGB order
                frames_batches = (results[0] * 255.).numpy().astype(np.uint8)
                frames = frames_batches.reshape(-1, *frames_batches.shape[-3:])
                label = data['label'].squeeze(-1)[0]
                cur_filename = cur_dirname.replace("test", "{:03d}_idx_{}".format(label, indices[i]))
                if i%100 == 0:
                    print(i, cur_filename)

                frame_list = list(frames)
                frame_list = _resize_frames(
                    frame_list,
                    args.target_resolution,
                    interpolation=args.resize_algorithm)

                video_clips = ImageSequenceClip(frame_list, fps=args.fps)
                out_type = osp.splitext(cur_filename)[1][1:]
                if out_type == 'gif':
                    video_clips.write_gif(cur_filename)
                else:
                    os.makedirs(os.path.dirname(cur_filename), exist_ok=True)
                    video_clips.write_images_sequence(cur_filename)
                    # video_clips.write_videofile(cur_filename, remove_temp=True)


    # Original pipeline
    # inputs = build_inputs(model, args.video, use_frames=args.use_frames)
    # gradcam = GradCAM(model, args.target_layer_name)
    # results = gradcam(inputs)

    # if args.out_filename is not None:
    #     try:
    #         from moviepy.editor import ImageSequenceClip
    #     except ImportError:
    #         raise ImportError('Please install moviepy to enable output file.')

    #     # frames_batches shape [B, T, H, W, 3], in RGB order
    #     frames_batches = (results[0] * 255.).numpy().astype(np.uint8)
    #     frames = frames_batches.reshape(-1, *frames_batches.shape[-3:])

    #     frame_list = list(frames)
    #     frame_list = _resize_frames(
    #         frame_list,
    #         args.target_resolution,
    #         interpolation=args.resize_algorithm)

    #     video_clips = ImageSequenceClip(frame_list, fps=args.fps)
    #     out_type = osp.splitext(args.out_filename)[1][1:]
    #     if out_type == 'gif':
    #         video_clips.write_gif(args.out_filename)
    #     else:
    #         video_clips.write_videofile(args.out_filename, remove_temp=True)


if __name__ == '__main__':
    main()
