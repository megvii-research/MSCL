"""
Flow提取算法：
mmaction, list of frames as input, (w, h, 3) -> (w, h, 2), no multi-processing.
usot, use gap to control every gap frames, adjacent(init as 4) to construct
3 frames as input to ARFlow. In fact, adjacent can adjust by flow scale.
"""
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import cv2
import pickle
from numpy.core.numeric import full
import torch
import numpy as np

from megaction.datasets import activity_dataset_configs
from meghair.utils.imgproc import imdecode
from meghair.utils import io
from datamaid2.storage.oss import OSS
from datamaid2.utils.logconf import get_logger
from torch.utils.data import Dataset, DataLoader
import refile
import nori2 as nori
from tqdm import tqdm

import rrun
import time
import concurrent.futures
from functools import partial
from collections import defaultdict
from tools.RAFT.inference import init_module

logger = get_logger(__name__)
class SimpleDataset(Dataset):
    def __init__(self, annos, transform, step=8, gap=2, adj=8) -> None:
        super().__init__()
        self.annos = annos
        self.fetcher = nori.Fetcher()
        self.transform = transform
        # Generate index region
        self.step = step
        self.new_annos = dict()
        idx = 0
        st_delta = step*gap  # start_index delta, e.g. 0, 16, 32
        sample_nums = (step-1)*gap + adj + 1    # max index -> (0, 14), so 15 frames [0:15]
        for anno in self.annos:
            imgs = anno['nori_id_seq']
            video_name = anno['video_name']
            for i, st_idx in enumerate(range(0, len(imgs), st_delta)):
                cur_imgs = imgs[st_idx:st_idx+sample_nums]
                cur_info = dict()
                cur_info['nori_id_seq'] = cur_imgs
                cur_info['video_name'] = video_name
                cur_info['index'] = i*step
                self.new_annos[idx] = cur_info
                idx += 1

    def __getitem__(self, idx):
        anno = self.new_annos[idx]

        imgs = anno['nori_id_seq']
        video_name = anno['video_name']
        index = anno['index']

        frames = [imdecode(self.fetcher.get(nid)) for nid in imgs]
        frames = [frame.astype(np.float32)[..., ::-1] for frame in frames]
        frames = [self.transform(img) for img in frames]
        return frames, video_name, index


    def __len__(self):
        return len(self.new_annos)

def flow_to_img(raw_flow, bound=20.):
    """Convert flow to gray image.

    Args:
        raw_flow (np.ndarray[float]): Estimated flow with the shape (w, h).
        bound (float): Bound for the flow-to-image normalization. Default: 20.

    Returns:
        np.ndarray[uint8]: The result list of np.ndarray[uint8], with shape
                        (w, h).
    """
    flow = np.clip(raw_flow, -bound, bound)
    flow += bound
    flow *= (255 / float(2 * bound))
    flow = flow.astype(np.uint8)
    return flow


def generate_flow(frames, method='tvl1'):
    """Estimate flow with given frames.

    Args:
        frames (list[np.ndarray[uint8]]): List of rgb frames, with shape
                                        (w, h, 3).
        method (str): Use which method to generate flow. Options are 'tvl1'
                    and 'farneback'. Default: 'tvl1'.

    Returns:
        list[np.ndarray[float]]: The result list of np.ndarray[float], with
                                shape (w, h, 2).
    """
    assert method in ['tvl1', 'farneback']
    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]

    if method == 'tvl1':
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()

        def op(x, y):
            return tvl1.calc(x, y, None)
    elif method == 'farneback':

        def op(x, y):
            return cv2.calcOpticalFlowFarneback(x, y, None, 0.5, 3, 15, 3, 5,
                                                1.2, 0)

    gray_st = gray_frames[:-1]
    gray_ed = gray_frames[1:]

    flow = [op(x, y) for x, y in zip(gray_st, gray_ed)]
    return flow


def extract_dense_flow(annos,
                       worker_id,
                       # ts,
                       dest_nori,
                       dest_meta,
                       gap=3, init_adjacent=4, step=16,
                       fetcher=None):
    """Extract dense flow given video or frames, save them as gray-scale
    images.

    Args:
        ts (TestHelper): TestHelper of flow method.
        imgs (list): Frames/noris the input video.
        dest (str): The directory to store the extracted flow images.
        video_name (str): The name of the input video.
        bound (float): Bound for the flow-to-image normalization. Default: 20.
        save_rgb (bool): Save extracted RGB frames. Default: False.
        start_idx (int): The starting frame index if use frames as input, the
            first image is path.format(start_idx). Default: 0.
        rgb_tmpl (str): The template of RGB frame names, Default:
            'img_{:05d}.jpg'.
        flow_tmpl (str): The template of Flow frame names, Default:
            '{}_{:05d}.jpg'.
        method (str): Use which method to generate flow. Options are 'tvl1'
            and 'farneback'. Default: 'tvl1'.
    """
    oss = OSS()
    ts = init_module()
    logger.info('start to process {}'.format(worker_id))

    nori_path = osp.join(dest_nori, f"{worker_id}.nori")
    wip_flag_path = os.path.splitext(nori_path)[0] + '.wip'
    if oss.exists(nori_path) and not oss.exists(wip_flag_path):
        logger.warning('skip existing nori: {}'.format(nori_path))
        return nori_path
    else:
        logger.warning('remove existing nori: {}'.format(nori_path))
        oss.delete_objects(nori_path)
        logger.warning('drop collection: {}'.format(nori_path))
        # db[collection_name].drop()
    oss.put_py_object('wip-flag', wip_flag_path)
    nr = nori.open(nori_path, 'w')

    # if fetcher is None:
    #     fetcher = nori.Fetcher()
    dataset = SimpleDataset(annos, transform=ts.input_transform, step=step, gap=gap, adj=init_adjacent)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=False)
    # for anno in tqdm(annos):
    nr_dict = {}
    for frames, video_name, cur_idx in tqdm(dataloader):
        video_name = video_name[0]   # For dataloader...
        cur_idx = cur_idx[0].item()
        logger.info(f"Start video {video_name}")
        # e.g. ['label', 'label_str', 'nori_id_seq', 'video_name']
        # Save two parts: nori + pkl(video_name -> nori)
        # imgs = anno['nori_id_seq']
        # video_name = anno['video_name']
        
        # frames = [imdecode(fetcher.get(nid)) for nid in imgs]
        # frames = [frame.astype(np.float32)[..., ::-1] for frame in frames]

        # list of ndarrary(cpu, float32)
        with torch.no_grad():
            flows = ts.inference_flows(frames, gap=gap, init_adjacent=init_adjacent, transform=False)
        # e.g. .../flow_raft/nori
        
        info_dict = {"flows":{}}
        assert len(flows) <= step, f"{len(flows)} vs {step}"
        for i in range(len(flows)):
            nori_name = '{}_flows_{:05d}'.format(video_name, i+cur_idx)
            nr_id = nr.put(flows[i].dumps(), filename=nori_name)
            info_dict["flows"][i+cur_idx] = nr_id

        # delete empty videos
        if len(info_dict["flows"]) == 0:
            logger.warning('video {} index {} frame nums is ZERO! skip...'.format(video_name, cur_idx))
        else:
            if video_name not in nr_dict:
                nr_dict[video_name] = info_dict
            else:
                len_before, len_cur = len(nr_dict[video_name]["flows"]), len(info_dict["flows"])
                nr_dict[video_name]["flows"].update(info_dict["flows"])
                len_after = len(nr_dict[video_name]["flows"])
                assert len_after == len_before + len_cur, f"{len_before} + {len_cur} vs {len_after}"
        torch.cuda.empty_cache()
    nr.close()
    
    # dict_path = args.local_pkl_path
    # os.makedirs(os.path.dirname(dict_path), exist_ok=True)
    # pickle.dump(nr_dict, open(dict_path, 'wb'))

    meta_path = osp.join(dest_meta, f"{worker_id}.pkl")
    oss.put_py_object(nr_dict, meta_path)

    if wip_flag_path.startswith('s3://'):
        oss.delete_object(wip_flag_path)
    else:
        os.remove(wip_flag_path)
    logger.info('finish worker id {}'.format(worker_id))

    return worker_id


def parse_args():
    parser = argparse.ArgumentParser(description='Extract flow and RGB images')
    parser.add_argument(
        '--dataset', default='kinetics155',
        help='dataset name')
    parser.add_argument('--prefix',
        default='',
        help='the prefix of input '
        'videos, used when input is a video list')
    parser.add_argument(
        '--dest',
        default='',
        help='the destination to save '
        'extracted frames')
    parser.add_argument(
        '--method',
        default='raft',
        help='use which method to '
        'generate flow, optional: [raft, arflow]')
    parser.add_argument(
        '--bound', type=float, default=20, help='maximum of '
        'optical flow')
    parser.add_argument(
        '--gid', type=int, default=0, help='maximum of '
        'optical flow')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Globlal size
    global_size = 3
    # -------------
    args = parse_args()
    video_paths=activity_dataset_configs[args.dataset]["dataset_path"]
    assert args.method in ['raft', 'arflow']
    dest = osp.join("s3://activity-public", args.dataset, f"flow_{args.method}ur")
    # 2 directories: noris/pkls e.g. noris -> train/val -> {worker_id}.nori -> nr.put(...)
    # if args.method == 'raft':
    #     from tools.RAFT.inference import init_module
    # elif args.method == 'arflow':
    #     from tools.ARFlow.inference import init_module
    # else:
    #     raise NotImplementedError(f"Not support method {args.method}")
    # ts = init_module()

    # Save -> 放在一个noris目录下即可
    # Hyper Parameters
    # (h, w) = [384, 640]
    worker_num = 16
    gap = 2
    init_adjacent = 8
    dest_nori = osp.join(dest, "noris", "{}")
    dest_meta = osp.join(dest, "pkls", "{}")
    # Simple test
    # worker_num = 1
    # for split in video_paths.keys():

    #     cur_pkl_path = video_paths[split]
    #     # TODO: split by workers
    #     with refile.smart_load_from(cur_pkl_path) as f:
    #         annos = io.load(f)
    #         step = len(annos)//worker_num
    #         for worker_id in range(worker_num):
                
    #             cur_annos = annos[worker_id*step:(worker_id+1)*step]
                
    #             extract_dense_flow(cur_annos, worker_id+11,
    #                 ts, dest_nori.format(split), dest_meta.format(split),
    #                 gap=gap, init_adjacent=init_adjacent)
    # _ = input()


    spec = rrun.RunnerSpec()
    spec.name = 'video2nori'
    log_dir = os.path.join('/data/datasets/rrun_log', '{}__{}'.format(
        os.path.basename(__file__), time.strftime('%Y-%m-%d.%H-%M-%S', time.localtime())))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    spec.log_dir = log_dir  # If you want to get logs of runners for debugging.
    spec.scheduling_hint.group = 'users'
    spec.charged_group = "research_video"
    spec.priority = "Low"
    spec.resources.cpu = 16
    spec.resources.gpu = 1
    spec.preemptible = False
    spec.resources.memory_in_mb = 40 * 1024
    spec.max_wait_time = 3600 * int(1e9)
    for split in video_paths.keys():

        cur_pkl_path = video_paths[split]
        # TODO: split by workers
        with refile.smart_load_from(cur_pkl_path) as f:
            full_annos = io.load(f)
            node_steps = (len(full_annos) + global_size)//global_size
            annos = full_annos[args.gid*node_steps: (args.gid+1)*node_steps]

            step = (len(annos) + worker_num)//worker_num

            pbar = tqdm(total=worker_num, unit="labels")
            worker = extract_dense_flow
            # worker = partial(extract_dense_flow, ts=ts, dest_nori=dest_nori.format(split), 
            # dest_meta=dest_meta.format(split), gap=gap, init_adjacent=init_adjacent)
            base_worker_id = args.gid*worker_num
            with rrun.RRunExecutor(spec, min(worker_num, 64)) as executor:
                futures = [
                    executor.submit(worker, annos[worker_id*step:(worker_id+1)*step], base_worker_id+worker_id, 
                                    dest_nori.format(split), dest_meta.format(split), gap, init_adjacent) 
                    for worker_id in range(worker_num)]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        idx = future.result()
                        logger.info('finish {}'.format(idx))
                        pbar.update(1)
                    except Exception as e:
                        # Catch remote exception
                        print(e)
            # for worker_id in range(worker_num):
                
            #     cur_annos = annos[worker_id*step:(worker_id+1)*step]
            #     print(worker_id, len(cur_annos), worker_id*step, (worker_id+1)*step)
                
                # extract_dense_flow(cur_annos, worker_id,
                #     ts, dest_nori.format(split), dest_meta.format(split),
                #     gap=gap, init_adjacent=init_adjacent)
