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

def merge_pkls(ori_anno_path, path_tmpl, tgt_path, worker_num=None, flow_shape=[384, 640]):
    oss = OSS()

    pkl_set = set(refile.smart_glob(path_tmpl.format('*')))
    ori_anno = load_anno(ori_anno_path)
    print(f"Find {len(pkl_set)} pkls!")
    if worker_num is None:
        worker_num = len(pkl_set)
    pkl_list = [path_tmpl.format(i) for i in range(worker_num)]
    print("============ Check with worker numbers ==============")
    print(pkl_set&set(pkl_list), pkl_set-set(pkl_list), set(pkl_list)-pkl_set, sep='\n')
    # It is worth nothing that this rely on ordered attribute in python dict
    results = [load_anno(ph) for ph in pkl_list]
    new_anno = []
    for i in range(len(ori_anno)):
        for j in range(len(results)):
            if ori_anno[i]['video_name'] in results[j]:
                ori_anno[i].update(results[j][ori_anno[i]['video_name']])
                new_anno.append(ori_anno[i])
    print(f"{len(new_anno)} vs {len(ori_anno)}")
    merged_pkl = new_anno
    for i in range(len(merged_pkl)):
        for key in ["enc_flows", "imflows"]:
            merged_pkl[i][key] = {int(k):v for k, v in merged_pkl[i][key].items()}

    h, w = flow_shape
    for i in range(len(merged_pkl)):
        if 'flows' in merged_pkl[i]:
            # Delete unencoded flows
            merged_pkl[i].pop('flows')
        if 'bboxs' in merged_pkl[i]:
            # Adjust box to relative scale
            bboxs = [[box[0]/w, box[1]/h, box[2]/w, box[3]/h] for box in merged_pkl[i]['bboxs']]
            if len(bboxs):
                assert max(map(max, bboxs)) <= 1, f"{max(map(max, bboxs))}"
            else:
                print("Find empty")
            merged_pkl[i]['bboxs'] = bboxs
    print(f"Merged length {len(merged_pkl)}")
    print(f"Save to {tgt_path}!")
    print(merged_pkl[0].keys())
    print(merged_pkl[0])
    vname_list = [meta['video_name'] for meta in merged_pkl]
    print("Different video names: ", len(set(vname_list)))
    # with open(f'/data/tmp/{osp.basename(tgt_path)}', 'wb') as f:
    #     pickle.dump(merged_pkl, f)
    oss.put_py_object(merged_pkl, tgt_path)
    print("Finished")

def check_bboxs(annos):
    from collections import defaultdict
    cs, cd = 0, 0
    bboxs, imflows = None, None
    for anno in annos:
        bboxs = anno['bboxs']
        imgs = anno['nori_id_seq']
        
        lb, lf = len(bboxs), len(imgs)
        if lb == lf:
            cs += 1
        else:
            print(lb, lf)
            cd += 1
    print(cs, cd)
    print(type(bboxs), type(imflows))


if __name__ == '__main__':
    # anno = load_anno("s3://activity-public/kinetics155/flow_raftur/annos/tmp/kinetics155_full_train_0.pkl")
    # print(len(anno), type(anno))
    # imflows = anno[0]['imflows']
    # print(len(imflows), type(imflows))
    # fetcher = nori.Fetcher()
    # _ = input()
    ori_t = "s3://activity-public/kinetics600/annos/kinetics400_new_train.pkl"
    ori_v =  "s3://activity-public/kinetics600/annos/kinetics400_new_val.pkl"
    tmpl_t = "s3://activity-public/kinetics400/flow_raft_ur/pkls/train/{}.pkl"
    tmpl_v = "s3://activity-public/kinetics400/flow_raft_ur/pkls/val/{}.pkl"
    tgt_t = "s3://activity-public/kinetics400/flow_raft_ur/annos_v2/kinetics400_full_train.pkl"
    tgt_v = "s3://activity-public/kinetics400/flow_raft_ur/annos_v2/kinetics400_full_val.pkl"

    anno_path_t = "s3://activity-public/kinetics155/flow_raftur/annos/kinetics155_with_flow_train.pkl"
    anno_path_v = "s3://activity-public/kinetics155/flow_raftur/annos/kinetics155_with_flow_val.pkl"
    nori_path_t = "s3://activity-public/kinetics155/flow_raftur/flow_noris/train"
    nori_path_v = "s3://activity-public/kinetics155/flow_raftur/flow_noris/val"
    p_num = 12
    # Code for merge pkls
    test_anno = load_anno("s3://activity-public/kinetics155/flow_raftur/annos/kinetics155_full_train_v3.pkl")
    print(test_anno[0])
    _ = input()

    merge_pkls(ori_t, tmpl_t, tgt_t)
    merge_pkls(ori_v, tmpl_v, tgt_v)
    _ = input()

    # For debug
    # annos_t = load_anno(tgt_t)
    # check_bboxs(annos_t)
    annos_v = load_anno(tgt_v)
    check_bboxs(annos_v)
    _ = input()


    # spec = rrun.RunnerSpec()
    # spec.name = 'video2nori'
    # log_dir = os.path.join('/data/datasets/rrun_log', '{}__{}'.format(
    #     os.path.basename(__file__), time.strftime('%Y-%m-%d.%H-%M-%S', time.localtime())))
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)

    # spec.log_dir = log_dir  # If you want to get logs of runners for debugging.
    # spec.scheduling_hint.group = 'users'
    # spec.charged_group = "research_video"
    # spec.priority = "Low"
    # spec.resources.cpu = 4
    # spec.resources.gpu = 0
    # spec.preemptible = False
    # spec.resources.memory_in_mb = 40 * 1024
    # spec.max_wait_time = 3600 * int(1e9)

    from multiprocessing import Process, Pool
    oss = OSS()
    meta_path = osp.join(osp.dirname(anno_path_t), "tmp")
    meta_path_tmpl_t = osp.join(meta_path, "kinetics155_full_train_v3_{}.pkl")
    print(meta_path_tmpl_t, "meta_path_tmpl_t")
    annos_t = load_anno(anno_path_t)    # list of dict

    annos_v = load_anno(anno_path_v)
    meta_path_tmpl_v = osp.join(meta_path, "kinetics155_full_val_v3_{}.pkl")
    print(meta_path_tmpl_v, "meta_path_tmpl_v")

    p = Pool(p_num)

    step_v = (len(annos_v)+p_num)//p_num
    # assert step_v*p_num >= len(annos_v)
    # pbar = tqdm(total=p_num, unit="labels")
    # with rrun.RRunExecutor(spec, min(p_num, 64)) as executor:
    #     futures = [
    #         executor.submit(process_flow, annos_v[wid*step_v:(wid+1)*step_v], nori_path_v, meta_path_tmpl_v, wid) 
    #         for wid in range(p_num)]
    #     for future in concurrent.futures.as_completed(futures):
    #         try:
    #             idx = future.result()
    #             logger.info('finish {}'.format(idx))
    #             pbar.update(1)
    #         except Exception as e:
    #             # Catch remote exception
    #             print(e)

    for wid in range(p_num):
        cur_annos = annos_v[wid*step_v:(wid+1)*step_v]
        p.apply_async(process_flow, args=(cur_annos, nori_path_v, meta_path_tmpl_v, wid))
    p.close()      
    p.join()  
    print("=== Finished val owo ===")

    p = Pool(p_num)
    step_t = (len(annos_t)+p_num)//p_num
    assert step_t*p_num >= len(annos_t)
    assert step_t > step_v, f"{step_t} vs {step_v}"
    # pbar = tqdm(total=p_num, unit="labels")
    # with rrun.RRunExecutor(spec, min(p_num, 64)) as executor:
    #     futures = [
    #         executor.submit(process_flow, annos_t[wid*step_t:(wid+1)*step_t], nori_path_t, meta_path_tmpl_t, wid) 
    #         for wid in range(p_num)]
    #     for future in concurrent.futures.as_completed(futures):
    #         try:
    #             idx = future.result()
    #             logger.info('finish {}'.format(idx))
    #             pbar.update(1)
    #         except Exception as e:
    #             # Catch remote exception
    #             print(e)

    for wid in range(p_num):
        cur_annos = annos_t[wid*step_t:(wid+1)*step_t]
        p.apply_async(process_flow, args=(cur_annos, nori_path_t, meta_path_tmpl_t, wid))
    p.close()      
    p.join()    
    
    merge_pkls(tmpl_t, tgt_t, p_num)
    merge_pkls(tmpl_v, tgt_v, p_num)
    anno = load_anno(tgt_v)
    print("Test: ", anno[0])
    
    print("=== Finished owo ===")
    # # TODO: Generate boxes, as a plugin module, saved at different files.