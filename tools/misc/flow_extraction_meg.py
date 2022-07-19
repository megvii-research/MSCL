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
import imghdr
import msgpack
import lz4.frame
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
from tools.RAFT.core.utils import flow_viz

logger = get_logger(__name__)
class SimpleDataset(Dataset):
    def __init__(self, cur_pkl_path, worker_num, worker_id,
                 transform, num_flow=8, gap=2, adj=8) -> None:
        super().__init__()
        with refile.smart_load_from(cur_pkl_path) as f:
            annos = io.load(f)

        step = (len(annos) + worker_num)//worker_num
        self.annos = annos[worker_id*step:(worker_id+1)*step]
        self.fetcher = nori.Fetcher()
        self.transform = transform
        # Generate index region
        self.num_flow = num_flow
        self.new_annos = dict()
        idx = 0
        st_delta = num_flow*gap  # start_index delta, e.g. 0, 16, 32
        sample_nums = (num_flow-1)*gap + adj + 1    # max index -> (0, 14), so 15 frames [0:15]
        for anno in self.annos:
            imgs = anno['nori_id_seq']
            video_name = anno['video_name']
            for i, st_idx in enumerate(range(0, len(imgs), st_delta)):
                cur_imgs = imgs[st_idx:st_idx+sample_nums]
                cur_info = dict()
                cur_info['nori_id_seq'] = cur_imgs
                cur_info['video_name'] = video_name
                cur_info['index'] = i*num_flow
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

def imencode(fmt, img, params=None):
    """\
    Encode a numpy array to image bytes.
    if fmt is '.np4', use msgpack and lz4 to compress image
    """
    if fmt == '.np4':
        compress_level = params
        if compress_level is None:
            compress_level = 3
        p = {'t': img.dtype.str, 's': img.shape, 'd': img.tobytes(order='C')}
        buf = lz4.frame.compress(
            msgpack.packb(p), compression_level=compress_level,
            block_size=lz4.frame.BLOCKSIZE_MAX1MB,
        )
        return 0, buf

    state, arr = cv2.imencode(fmt, img, params)
    return state, arr.tobytes()

def imdecode(buf, flags=cv2.IMREAD_UNCHANGED):
    """\
    Decode an ordinaray or np4 image content to numpy array
    """
    if imghdr.what('', h=buf) is not None:
        return cv2.imdecode(np.frombuffer(buf, np.uint8), flags)

    try:
        pb = lz4.frame.decompress(buf)
        p = msgpack.unpackb(pb)
    except Exception:
        return None

    return np.frombuffer(p['d'], dtype=p['t']).reshape(p['s'])


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


def extract_dense_flow(cur_pkl_path,
                       base_worker_num,
                       worker_id,
                       dest_nori,
                       dest_meta,
                       split_num=30, gap=3, init_adjacent=4, num_flow=16,
                       fetcher=None):
    oss = OSS()
    ts = init_module()
    logger.info('start to process {}'.format(worker_id))
    

    worker_num = base_worker_num*split_num
    logger.info(f"Current split_ids: {list(range(worker_id*split_num, (worker_id+1)*split_num))}")
    # for split_id in range(worker_id*split_num, (worker_id+1)*split_num):
    for split_id in range(worker_id, worker_num, base_worker_num):
        nori_path = osp.join(dest_nori, f"{split_id}.nori")
        wip_flag_path = os.path.splitext(nori_path)[0] + '.wip'
        if oss.exists(nori_path) and not oss.exists(wip_flag_path):
            logger.warning('skip existing nori: {}'.format(nori_path))
            continue
        else:
            logger.warning('remove existing nori: {}'.format(nori_path))
            oss.delete_objects(nori_path)
            logger.warning('drop collection: {}'.format(nori_path))
            # db[collection_name].drop()
        oss.put_py_object('wip-flag', wip_flag_path)
        nr = nori.open(nori_path, 'w')

        # if fetcher is None:
        #     fetcher = nori.Fetcher()
        dataset = SimpleDataset(cur_pkl_path, worker_num, split_id, 
            transform=ts.input_transform, num_flow=num_flow, gap=gap, adj=init_adjacent)
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
            
            info_dict = {"enc_flows":{}, "imflows":{}}
            assert len(flows) <= num_flow, f"{len(flows)} vs {num_flow}"
            for i in range(len(flows)):
                cflow = flows[i]
                fid = i+cur_idx
                # Imflows
                cflow_rgb = flow_viz.flow_to_image(cflow, convert_to_bgr=True)
                nori_name = '{}_imflows_{:05d}'.format(video_name, fid)
                st, enc_jpg = cv2.imencode(".jpg", cflow_rgb)
                nr_id = nr.put(enc_jpg.tobytes(), filename=nori_name)
                info_dict["imflows"][i+cur_idx] = nr_id
                # Flows
                st, enc_arr = imencode('.np4', cflow)       # 0, buf
                nori_name_flow = '{}_flows_{:05d}'.format(video_name, fid)
                nr_id_flow = nr.put(enc_arr, filename=nori_name_flow)
                info_dict["enc_flows"][i+cur_idx] = nr_id_flow

            # delete empty videos
            if len(info_dict["imflows"]) == 0:
                logger.warning('video {} index {} frame nums is ZERO! skip...'.format(video_name, cur_idx))
            else:
                if video_name not in nr_dict:
                    nr_dict[video_name] = info_dict
                else:
                    len_before, len_cur = len(nr_dict[video_name]["imflows"]), len(info_dict["imflows"])
                    nr_dict[video_name]["imflows"].update(info_dict["imflows"])
                    nr_dict[video_name]["enc_flows"].update(info_dict["enc_flows"])
                    len_after = len(nr_dict[video_name]["imflows"])
                    assert len_after == len_before + len_cur, f"{len_before} + {len_cur} vs {len_after}"
            torch.cuda.empty_cache()
        nr.close()
        
        # dict_path = args.local_pkl_path
        # os.makedirs(os.path.dirname(dict_path), exist_ok=True)
        # pickle.dump(nr_dict, open(dict_path, 'wb'))

        meta_path = osp.join(dest_meta, f"{split_id}.pkl")
        oss.put_py_object(nr_dict, meta_path)

        if wip_flag_path.startswith('s3://'):
            oss.delete_object(wip_flag_path)
        else:
            os.remove(wip_flag_path)
        logger.info('finish worker id {} split id {}'.format(worker_id, split_id))

    return worker_id


def parse_args():
    parser = argparse.ArgumentParser(description='Extract flow and RGB images')
    parser.add_argument(
        '--dataset', default='ucf101',
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
    # x = np.arange(1000).reshape((100, 10)).astype(np.uint8)
    # _, y = imencode('.np4', x)
    # print(y, imdecode(y), type(y))
    # import sys
    # print(sys.getsizeof(x), sys.getsizeof(y))
    # _ = input()
    # Globlal size
    # -------------
    args = parse_args()
    video_paths=activity_dataset_configs[args.dataset]["dataset_path"]
    # Hyper Parameters
    # (h, w) = [384, 640]
    worker_num = 8
    gap = 2
    init_adjacent = 4
    split_num=30

    dataset_map = dict(ucf101='UCF101', kinetics400='kinetics400')
    assert args.method in ['raft', 'arflow']
    dest = osp.join("s3://activity-public", dataset_map[args.dataset], f"flow_{args.method}_ur_{init_adjacent}")

    # Save -> 放在一个noris目录下即可
    dest_nori = osp.join(dest, "noris", "{}")
    dest_meta = osp.join(dest, "pkls", "{}")
    # Simple test
    # worker_num = 8
    # for split in video_paths.keys():

    #     cur_pkl_path = video_paths[split]
    #     # TODO: split by workers
    #     # with refile.smart_load_from(cur_pkl_path) as f:
    #     #     annos = io.load(f)
    #     #     step = len(annos)//worker_num
    #     for worker_id in range(worker_num):            
    #         extract_dense_flow(cur_pkl_path, worker_num, worker_id,
    #             dest_nori.format(split), dest_meta.format(split),
    #             gap=gap, init_adjacent=init_adjacent)
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
    spec.resources.cpu = 6
    spec.resources.gpu = 1
    spec.preemptible = False
    spec.resources.memory_in_mb = 40 * 1024
    spec.max_wait_time = 3600 * int(1e9)
    for split in video_paths.keys():
        if split == 'val':
            continue

        cur_pkl_path = video_paths[split]
        # TODO: split by workers
        pbar = tqdm(total=worker_num, unit="labels")
        worker = extract_dense_flow
        # worker = partial(extract_dense_flow, ts=ts, dest_nori=dest_nori.format(split), 
        # dest_meta=dest_meta.format(split), gap=gap, init_adjacent=init_adjacent)
        with rrun.RRunExecutor(spec, min(worker_num, 64)) as executor:
            futures = [
                executor.submit(worker, cur_pkl_path, worker_num, worker_id,
                                dest_nori.format(split), dest_meta.format(split), split_num, gap, init_adjacent) 
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
