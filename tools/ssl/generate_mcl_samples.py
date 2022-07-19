# Temporal Sampling
from scipy.ndimage.filters import sobel
from mmaction.utils import imdecode
import nori2 as nori
import numpy as np
import refile
import cv2
import os
import torch
import kornia
import argparse
import torch.nn.functional as F
from meghair.utils import io
from scipy import ndimage, misc
from datamaid2.storage.oss import OSS
from multiprocessing import Pool

from tools.RAFT.core.utils import flow_viz

def cal_motion_map(flow):
    # flow: ndarray with h, w, 2
    u, v = flow[..., 0], flow[..., 1]

    # sobel_u_x = cv2.Sobel(u, -1, 1, 0, ksize=3)   # need to be h,w,c
    sobel_u_x = ndimage.sobel(u, axis=-1)   # h,w is ok
    sobel_u_y = ndimage.sobel(u, axis=0)
    sobel_v_x = ndimage.sobel(v, axis=-1)
    sobel_v_y = ndimage.sobel(v, axis=0)
    motion_map = np.sqrt(np.square(sobel_u_x)+np.square(sobel_u_y)+np.square(sobel_v_x)+np.square(sobel_v_y))   # h,w
    
    return motion_map

def cal_attention_map(mp, att_type='max'):
    sl = 28
    cmp = torch.from_numpy(mp).unsqueeze(0).unsqueeze(0).float()   # 1,1,h,w
    cmp_hw = cmp.shape[-2:]
    cmp = F.avg_pool2d(cmp, kernel_size=(sl, sl), stride=(sl, sl))
    cmp = F.interpolate(cmp, size=cmp_hw, mode='bilinear')

    cmp = cmp[0, 0]
    if att_type == 'max':
        cmp = cmp/cmp.max()
    elif att_type == 'sum':
        cmp = cmp/cmp.sum()
    else:
        raise ValueError(f"Not find type {att_type}!")
    
    return cmp

def cal_rgb_map(flow, att_type='none'):

    cflow_rgb = flow_viz.flow_to_image(flow, convert_to_bgr=False).astype(np.float32)   # h,w,3
    # TODO: use CIEDE2000, e.g. https://zh.wikipedia.org/wiki/%E9%A2%9C%E8%89%B2%E5%B7%AE%E5%BC%82

    if att_type == 'none':
        return cflow_rgb
    elif att_type in ['max', 'sum']:
        motion_map = cal_motion_map(flow)
        att = cal_attention_map(motion_map, att_type=att_type).unsqueeze(-1).numpy()    # h,w,1
        return att*cflow_rgb
    else:
        raise ValueError(f"Not find type {att_type}")

def process_single_flow(flow, weight_type, att_type='none'):
    if weight_type == 'motion_map':
        weight = cal_motion_map(flow)
    elif weight_type == 'attention_map':
        motion_map = cal_motion_map(flow)
        weight = cal_attention_map(motion_map, att_type=att_type)
    elif weight_type == 'rgb_map':
        weight = cal_rgb_map(flow, att_type=att_type)
    else:
        raise ValueError(f"Not find {weight_type}")
    return weight

def process_video(meta, weight_type, att_type, pool_type='avg', clip_len=8, clip_stride=4):
    """
    返回索引数组，random采样时根据上界确定索引数组上界，并从数组中随机选择clip_offset.
    3种weight类型：motion/attention(motion + down-up sample)/rgb
    """

    if pool_type == 'avg':
        pool_func = lambda x: x.mean((0, 1))
    elif pool_type == 'max':
        pool_func = lambda x: x.max((0, 1))
    else:
        raise ValueError(f"{pool_type} is not supported.")

    nf = nori.Fetcher()
    video_weights = []
    if isinstance(meta['enc_flows'], dict):
        for idx in range(len(meta['enc_flows'])):
            nid = meta['enc_flows'][idx]
            flow = imdecode(nf.get(nid))
            weight = process_single_flow(flow, weight_type=weight_type, att_type=att_type)
            video_weights.append(weight)
    else:
        for nid in meta['enc_flows']:
            flow = imdecode(nf.get(nid))
            weight = process_single_flow(flow, weight_type=weight_type, att_type=att_type)
            video_weights.append(weight)

    if 'rgb' in weight_type:
        # To Diff
        new_video_weights = []
        video_weights.append(video_weights[-1])
        for i in range(len(video_weights)-1):
            cur = video_weights[i] - video_weights[i+1]     # h,w,c
            cur = np.linalg.norm(cur, axis=-1)  # h,w
            new_video_weights.append(cur)
        video_weights = new_video_weights

    # suppose out_of_bound_opt='loop'
    vid_len = len(video_weights)
    for i in range(vid_len):
        assert video_weights[i].ndim == 2, f"Number of dims is {video_weights[i].ndim}"
        video_weights[i] = pool_func(video_weights[i])
    clip_weights = []
    for i in range(vid_len):
        cur = 0
        for j in range(clip_len):
            # Because in training, out_of_bound samples is rarely when frame number is enough
            if i+j*clip_stride < vid_len:
                cur += video_weights[i+j*clip_stride]
        cur /= clip_len
        clip_weights.append(cur)

    clip_median = np.median(clip_weights)
    rt = []
    for i, v in enumerate(clip_weights):
        if v > clip_median:
            rt.append(i)
    meta['chosen_idx'] = rt
    return meta

def process_anno(anno_items):
    anno, worker_id, new_path_tmpl = anno_items
    oss = OSS()
    # Train
    print(f"Start {worker_id} with size {len(anno)}")
    for i in range(len(anno)):
        if i%10 == 0:
            print(f"Worker id: {worker_id}: idx {i}, videoname {anno[i]['video_name']}")
        anno[i] = process_video(
            anno[i], weight_type=args.weight_type, att_type=args.att_type,
            clip_len=args.clip_len, clip_stride=args.clip_stride)
    oss.put_py_object(anno, new_path_tmpl.format(worker_id))
    
def load_anno(path):
    with refile.smart_load_from(path) as f:
        annos = io.load(f)
    return annos

def parse_args():
    parser = argparse.ArgumentParser(description='MoCo temporal index choosen')

    parser.add_argument('--clip-len', type=int, default=8, help='clip_len')
    parser.add_argument('--clip-stride', type=int, default=4, help='clip_stride')
    parser.add_argument('--num-workers', type=int, default=1, help='num_workers')
    parser.add_argument(
        '--weight-type', type=str, default='motion_map', help='weight_type')
    parser.add_argument(
        '--att-type', type=str, default='max', help='att_type')
    parser.add_argument(
        '--pth-t', type=str, 
        default='s3://activity-public/kinetics155/flow_raftur/annos/kinetics155_full_train_v3.pkl', 
        help='pth_t')
    parser.add_argument(
        '--pth-v', type=str, 
        default='s3://activity-public/kinetics155/flow_raftur/annos/kinetics155_full_val_v3.pkl', 
        help='pth_v')
    parser.add_argument(
        '--out-dir', type=str, 
        default='s3://activity-public/kinetics155/flow_raftur/annos/new', 
        help='out_dir')

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    weight_map = dict(
        motion_map='mm', attention_map='am', rgb_map='rm')
    att_map = dict(
        none='ne', max='mx', sum='sm')
    # Hyper parameters
    # weight_type: motion_map, attention_map, rgb_map
    # att_type: none, max, sum      # none only for rgb_map
    # pool_type: max, avg
    # load annos
    # anno_t = load_anno(pth_t)

    args = parse_args()
    out_dir = args.out_dir
    ori_name_t = os.path.basename(args.pth_t)
    ori_name_v = os.path.basename(args.pth_v)
    base_new_name = '_' + weight_map[args.weight_type] + '_' + att_map[args.att_type] + f'_s{args.clip_stride}' 
    new_name_t = ori_name_t.replace('.', base_new_name+'_{:03d}'+'.')
    new_name_v = ori_name_v.replace('.', base_new_name+'_{:03d}'+'.')
    new_path_tmpl_t = os.path.join(out_dir, new_name_t)
    new_path_tmpl_v = os.path.join(out_dir, new_name_v)

    print(args, new_path_tmpl_t, new_path_tmpl_v)
    print(f"Please check: stride {args.clip_stride}, len {args.clip_len}")

    def process_single_part(pth, num_workers, new_path_tmpl):
        oss = OSS()
        # Train
        print(f"Start {pth}")
        cur_anno = load_anno(pth)
        cur_size = len(cur_anno)
        cur_stride = (cur_size+num_workers-1)//num_workers
        print(f"Stride {cur_stride}")

        anno_list = [cur_anno[i*cur_stride: (i+1)*cur_stride] for i in range(num_workers)]
        worker_id_list = list(range(num_workers))
        new_path_tmpl_list = [new_path_tmpl for _ in range(num_workers)]
        pool = Pool(num_workers)
        pool.map(
            process_anno,
            zip(anno_list, worker_id_list, new_path_tmpl_list,) )
        pool.close()
        pool.join()

    process_single_part(args.pth_t, args.num_workers, new_path_tmpl_t)
    # process_single_part(args.pth_v, args.num_workers, new_path_tmpl_v)
