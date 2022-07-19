import os.path as osp
import os
from nori2.v2 import Fetcher, index

import refile
import nori2 as nori
import numpy as np
import imghdr
import msgpack
import lz4.frame
import cv2
import copy
import pickle
import rrun
import time
import concurrent.futures
import sys
from simplejson import load
from tqdm import tqdm
from datamaid2.storage.oss import OSS
from meghair.utils import io
from meghair.utils.imgproc import imdecode
from torch.utils.data import Dataset, DataLoader

from tools.RAFT.flow_utils import flow_to_image, calc_corner_bbox_freq, \
    resize_flow, flow_to_bbox, smooth_bbox_dp, calc_nearby_bbox_freq
from tools.RAFT.core.utils import flow_viz
from datamaid2.utils.logconf import get_logger

logger = get_logger(__name__)
class SimpleDataset(Dataset):
    def __init__(self, flow_noris, fetcher) -> None:
        super().__init__()
        self.flow_noris = flow_noris
        self.fetcher = fetcher

    def __getitem__(self, idx):
        cflow = load_flow(self.flow_noris[idx], self.fetcher) 
        return cflow

    def __len__(self):
        return len(self.flow_noris)

def get_dataloader(flow_noris, fetcher):
    dataset = SimpleDataset(flow_noris, fetcher)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=1,
    )
    return dataloader

def inference_bboxs(im_length, flows, vis=False, gap=2, adjacent=8):
    # Note that cut_ratio is used to cut the margins of the flow map, as margin flows are always of low quality
    cut_ratio = 1/32
    # Convert flow map to candidate boxes (B)
    bboxs = [flow_to_bbox(flow, cut_ratio=cut_ratio) for flow in flows]
    # Use Dynamic Programming (DP) to generate reliable pseudo box sequences (B')
    bboxs, picked_frame_index, bbox_found_freq, bbox_picked_freq, aver_vary = \
        smooth_bbox_dp(bboxs, length=im_length, adjacent=adjacent, gap=gap)

    # Calc the bbox DP-select rate (bbox_freq) for every frame among all its adjacent frames
    # Note: search range is the frame interval for calculating frame quality (Denoted as T_s in the paper)
    # In practice, short interval (3) is better according to our experiments (10 is deprecated)
    freq_dict = calc_nearby_bbox_freq(picked_frame_index, video_length=len(bboxs),
                                    search_range=[3, 10], gap=gap)

    # The frequency of corner bboxes in the smoothed bbox sequence (B')
    # As an implementation detail, we actually give priority to sequences with less corner boxes (for center bias)
    corner_bbox_freq = calc_corner_bbox_freq(bboxs, img_shape=flows[0].shape, cut_ratio=cut_ratio)

    # Visualize optical flow
    # flow_vis = [flow_to_image(flow) for flow in flows]

    # if vis:
    #     i = 0
    #     for i in range(len(flows)):
    #         bbox = bboxs[2*i]
    #         # image = flow_viz.flow_to_image(flows[i])
    #         image = imgs[i].astype(np.uint8)
    #         image = cv2.resize(image, (640, 384))   # (w, h)
    #         text = "{:.2f}/{:.2f}".format(freq_dict[i][0], freq_dict[i][1])
    #         draw = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
    #                             (0, 255, 0), 1)
    #         draw = cv2.putText(draw, text, (int(bbox[0])+25, int(bbox[1])+25),
    #                         cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    #         tar = os.path.join("/home/nijingcheng/mmaction2/tools/RAFT/visualize", f"{i*gap}_box.png")
    #         cv2.imwrite(tar, draw*255)
    #         i += 1

    return bboxs, picked_frame_index, (freq_dict, bbox_found_freq, bbox_picked_freq, aver_vary, corner_bbox_freq)

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

def load_anno(path):
    with refile.smart_load_from(path) as f:
        annos = io.load(f)
    return annos

def load_anno_fin(path):
    with refile.smart_open(path, "rb") as f:
        data = pickle.load(f)
    return data

def load_pkls(path):
    pkl_list = refile.smart_glob(osp.join(path, "*.pkl"))
    print(f"Find {len(pkl_list)} pkls!")
    results = [load_anno(ph) for ph in pkl_list]
    return results

def load_flow(nid, fetcher):
    # loads: 从byte数据中读取
    flow = pickle.loads(fetcher.get(nid))
    return flow

def get_idmap(anno):
    # video_name to ids
    return {anno[i]['video_name']:i for i in range(len(anno))}

def add_flow_seq(anno, idmap, pkls):
    # keys in anno: ['label', 'label_str', 'nori_id_seq', 'video_name']
    new_anno = [None for _ in range(len(anno))]
    cal = 0
    for pt in pkls:
        for vname in pt:
            idx = idmap[vname]
            new_anno[idx] = copy.deepcopy(anno[idx])
            new_anno[idx]['flows'] = pt[vname]['flows']
            cal += 1
    print("Infos: ", cal, len(anno), len(list(filter(lambda x: x is None, new_anno))))
    for i in range(len(anno)):
        if new_anno[i] is None:
            print(anno[i])
    new_anno = list(filter(lambda x: x is not None, new_anno))
    return new_anno

def visualize_box(anno_path):
    fetcher = nori.Fetcher()
    annos = load_anno(anno_path)    # list of dict

    cur_anno = annos[0]
    flows = [load_flow(cur_anno['flows'][k], fetcher) for k in cur_anno['flows']]
    # nid -> bytes -> img
    frames = [imdecode(fetcher.get(nid)) for nid in cur_anno['nori_id_seq']]
    frames = [frame.astype(np.float32)[..., ::-1] for frame in frames]  # h, w, c, rgb
    print(len(flows), len(frames), "Length")
    inference_bboxs(frames, flows, gap=2, adjacent=8)

def process_flow(annos, base_nori_path, meta_path_tmpl, idx, saved_interval=100):
    # keys in anno: ['label', 'label_str', 'nori_id_seq', 'video_name', 'flows']
    meta_tgt = meta_path_tmpl.format(idx)
    if refile.smart_exists(meta_tgt):
        logger.info(f"Find path {meta_tgt} exist, skip!")
        return idx
    logger.info(f"Start worker {idx} with size {len(annos)}!")
    fetcher = nori.Fetcher()
    oss = OSS()
    new_annos = []
    vname_set = set()
    # Resume from part target
    part_meta_tgt = meta_path_tmpl.format(f"part_{idx}")
    if refile.smart_exists(part_meta_tgt):
        logger.info(f"Find path {part_meta_tgt} exist, continue!")
        new_annos = load_anno(part_meta_tgt)
        logger.info(f"Load with {len(new_annos)} annos")
        vname_set = set([meta['video_name'] for meta in new_annos])
        logger.info(f"Verify with {len(vname_set)} annos")
    nori_path = osp.join(base_nori_path, f"{idx}.nori")
    print("Nori path: ", nori_path)
    nr = nori.open(nori_path, 'w')
    pbar = tqdm(total=len(annos), unit=f'its_{idx}')    
    for i, meta in enumerate(annos):
        video_name = meta['video_name']
        if video_name in vname_set:
            pbar.update(1)
            continue
        new_meta = copy.deepcopy(meta)
        new_meta['imflows'] = []
        new_meta['enc_flows'] = []
        flow_noris = meta['flows']
        cflows = []
        # flow_loader = get_dataloader(flow_noris, fetcher)
        # for fid, cflow in enumerate(flow_loader):
        #     cflow = cflow[0].numpy()
        for fid in range(len(flow_noris)):
            cflow = load_flow(flow_noris[fid], fetcher)     # ndarray, (h, w, c), float32
            # * Convert to BGR
            cflow_rgb = flow_viz.flow_to_image(cflow, convert_to_bgr=True)       # ndarray, (h, w, c), uint8
            # Save to nori
            nori_name = '{}_imflows_{:05d}'.format(video_name, fid)
            # img encode
            # jpg is small than png (7x).
            st, enc_jpg = cv2.imencode(".jpg", cflow_rgb)
            # tobytes can use imdecode
            nr_id = nr.put(enc_jpg.tobytes(), filename=nori_name)
            new_meta['imflows'].append(nr_id)
            # ndarray encode
            st, enc_arr = imencode('.np4', cflow)       # 0, buf
            nori_name = '{}_flows_{:05d}'.format(video_name, fid)
            nr_id_flow = nr.put(enc_arr, filename=nori_name)
            new_meta['enc_flows'].append(nr_id_flow)
            cflows.append(cflow)
        assert len(new_meta['imflows']) == len(meta['flows']), \
            f"{len(new_meta['imflows'])} != {len(meta['flows'])}..."
        # Calculate bboxs
        try:
            bboxs, picked_frame_index, stat_tuple = inference_bboxs(
                len(meta["nori_id_seq"]), cflows, gap=2, adjacent=8,
            )
        except Exception as e:
            logger.info(f"Find {e}, skip!")
            bboxs = []
        new_meta["bboxs"] = bboxs
        new_annos.append(new_meta)
        pbar.update(1)
        # Save temp results
        if i%saved_interval == 0:
            logger.info(f"Save to {part_meta_tgt} with size {len(new_annos)}...")
            oss.put_py_object(new_annos, part_meta_tgt)
    assert len(new_annos) == len(annos), f"{len(new_annos)} != {len(annos)}..."
    print(f"Save to {meta_path_tmpl.format(idx)}...")
    oss.put_py_object(new_annos, meta_path_tmpl.format(idx))
    return idx

def generate_subset(ori_anno_path, tgt_anno_path, tgt_output_path):
    oss = OSS()

    print("Start Load")
    ori_anno = load_anno(ori_anno_path)
    print(f"Finish Load {ori_anno_path}")
    tgt_anno = load_anno(tgt_anno_path)  # with subset
    print(f"Finish Load {tgt_anno_path}")
    video_names = [anno['video_name'] for anno in tgt_anno]

    new_anno = []
    for i, anno in enumerate(ori_anno):
        print(i)
        if anno['video_name'] in video_names:
            new_anno.append(anno)
    print(f"{len(tgt_anno)} vs {new_anno}")
    oss.put_py_object(new_anno, tgt_output_path)
    
    print("Finished")

if __name__ == '__main__':
    tgt_out_t = "s3://activity-public/kinetics155/flow_raftur_16/annos/kinetics155_full_train.pkl"
    tgt_out_v =  "s3://activity-public/kinetics155/flow_raftur_16/annos/kinetics155_full_val.pkl"
    tgt_t = "s3://activity-public/kinetics155/flow_raftur/annos/kinetics155_full_train_v3.pkl"
    tgt_v = "s3://activity-public/kinetics155/flow_raftur/annos/kinetics155_full_val_v3.pkl"
    ori_t = "s3://activity-public/kinetics400/flow_raft_ur_16/annos/kinetics400_full_train.pkl"
    ori_v = "s3://activity-public/kinetics400/flow_raft_ur_16/annos/kinetics400_full_val.pkl"

    # Code for merge pkls
    generate_subset(ori_t, tgt_t, tgt_out_t)
    generate_subset(ori_v, tgt_v, tgt_out_v)

    # For debug
    # annos_t = load_anno(tgt_t)
    # check_bboxs(annos_t)
    # annos_v = load_anno(tgt_v)
    # check_bboxs(annos_v)
    # _ = input()
    
    print("=== Finished owo ===")
    # # TODO: Generate boxes, as a plugin module, saved at different files.