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
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from datamaid2.storage.oss import OSS
from meghair.utils import io
from meghair.utils.imgproc import imdecode
from scipy import ndimage, misc
from torchvision import transforms as transforms
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

    return np.frombuffer(p[b'd'], dtype=p[b't']).reshape(p[b's'])

def load_anno(path):
    with refile.smart_load_from(path) as f:
        annos = io.load(f)
    return annos

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

nf = nori.Fetcher()
def get_img_from_nid(nori_id, channel_order='rgb'):
    image = nf.get(nori_id)
    image = np.frombuffer(image, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    if channel_order == "rgb":
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
    return image

def parse_label(dataset_name):
    if dataset_name == "sthv2":
        oss_path = "s3://activity-public/something-something/somethingv2/something-something-v2-labels.json"
        with refile.smart_load_from(oss_path) as f:
            annos = json.load(f)
        label_map = {int(v):k for k, v in annos.items()}
    else:
        raise ValueError(f"Unrecognized dataset name {dataset_name}.")
    return label_map

def apply_transform(img, tfm):
    # img -> ndarray, hwc
    img = torch.tensor(img.copy()).permute(2, 0, 1)
    print(img.dtype)
    img = tfm(img)
    img = img.permute(1, 2, 0)
    if img.is_floating_point():
        img = img*255
    img = img.numpy().astype(np.uint8)
    return img

def flow_uv_to_colors(u: torch.Tensor, v, colorwheel, convert_to_bgr=False, div255=True):
    """
    Args:
        u (np.ndarray): Input horizontal flow of shape [B,H,W]
        v (np.ndarray): Input vertical flow of shape [B,H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    # 这部分操作与上下文无关的，简单来说，可以将其移到gpu上=
    flow_image = torch.zeros((u.shape[0], u.shape[1], u.shape[2], 3), dtype=torch.uint8, device=u.device)
    ncols = colorwheel.shape[0]
    rad = torch.sqrt(torch.square(u) + torch.square(v))
    print(rad)
    a = torch.atan2(-v, -u)/math.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = torch.floor(fk).to(torch.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    print(k0.max(), k0.min(), k1.max(), k1.min())
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        print(col0.max(), col0.min(), col1.max(), col1.min())
        col = (1-f)*col0 + f*col1
        print(col.max(), col.min())
        idx = (rad <= 1)
        print(col[idx], col[idx].max(), col[idx].min())
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        print(col[idx], col[idx].max(), col[idx].min())
        print(rad[idx], col[idx], idx.sum())
        # _ = input()
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[...,ch_idx] = torch.floor(255 * col)
    flow_image = flow_image.permute(0, 3, 1, 2).float()
    if div255:
        flow_image /= 255
    return flow_image

def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False, div=1):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)

    u = torch.from_numpy(u).unsqueeze(0)
    v = torch.from_numpy(v).unsqueeze(0)
    colorwheel = flow_viz.make_colorwheel()
    img = flow_uv_to_colors(u/div, v/div, colorwheel, convert_to_bgr=convert_to_bgr, div255=False)[0]
    img = img.permute(1, 2, 0)
    img = img.numpy().astype(np.uint8)
    print(img.shape, img.max(), img.min())
    return img

if __name__ == '__main__':
    anno_path = 's3://activity-public/something-something/annos/somethingv2/somethingv2_train.pkl'
    anno_path_v = 's3://activity-public/something-something/annos/somethingv2/somethingv2_val.pkl'
    tar_path = 's3://activity-public/kinetics155/somethingv2_train_part.pkl'
    tar_path_v = 's3://activity-public/kinetics155/somethingv2_val_part.pkl'
    re_list = ['can\'t', 'but', 'without', 'on', 'Pretend', 'just', 'slightly', 'to', 'like', 'and', 'with', 'over', 'show']

    # def choose(x):
    #     for r in re_list:
    #         if r.lower() in x[1].lower():
    #             return False
    #     return True
    
    # label_map = parse_label('sthv2')
    # lbs = label_map.items()
    # cal = 0
    # chosen = []
    # for idx, nm in filter(choose, lbs):
    #     cal += 1
    #     chosen.append(idx)
    #     print(nm)
    # print(len(chosen))
    # anno_v = load_anno(anno_path_v)
    # print(len(anno_v))
    # mxid = 0
    # new_v = []
    # for an in anno_v:
    #     cid = an['label']
    #     mxid = max(cid, mxid)
    #     if cid in chosen:
    #         new_v.append(an)
    # print(len(new_v))
    # new_t = []
    # anno_t = load_anno(anno_path)
    # for an in anno_t:
    #     cid = an['label']
    #     mxid = max(cid, mxid)
    #     if cid in chosen:
    #         new_t.append(an)
    # oss = OSS()
    # oss.put_py_object(new_t, tar_path)
    # oss.put_py_object(new_v, tar_path_v)

    from tools.misc.flow2img import imdecode
    anno_full_v = "s3://activity-public/kinetics155/flow_raftur/annos/kinetics155_full_val_v3.pkl"
    anno = load_anno(anno_full_v)
    tfm = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    for i in range(3):
        cur = i*8
        # img = get_img_from_nid(anno[0]['nori_id_seq'][cur], channel_order='bgr')
        enc_flow = imdecode(nf.get(anno[0]['enc_flows'][i*4]))
        mp = cal_motion_map(enc_flow)
        mp = cal_attention_map(mp).unsqueeze(-1)  # h,w
        mp = (mp*255).numpy().astype(np.uint8)
        tar = os.path.join("/home/nijingcheng/mmaction2/tools/RAFT/visualize", f"{i}_mp.png")
        cv2.imwrite(tar, mp)
        _ = input()
        # tar = os.path.join("/home/nijingcheng/mmaction2/tools/RAFT/visualize", f"{i}_img.png")
        # cv2.imwrite(tar, img)
        # tar = os.path.join("/home/nijingcheng/mmaction2/tools/RAFT/visualize", f"{i}_flow_ori_v2.png")
        # cv2.imwrite(tar, flow_img)
        # flow_img = flow_img[..., ::-1]
        # flow_img = apply_transform(flow_img, tfm)[..., ::-1]
        # tar = os.path.join("/home/nijingcheng/mmaction2/tools/RAFT/visualize", f"{i}_flow_tfm_v2.png")
        # cv2.imwrite(tar, flow_img)
        enc_flow = imdecode(nf.get(anno[0]['enc_flows'][i*4]))
        flow_img = flow_to_image(enc_flow*3, convert_to_bgr=True, div=1)
        tar = os.path.join("/home/nijingcheng/mmaction2/tools/RAFT/visualize", f"{i}_div1.png")
        print(tar)
        cv2.imwrite(tar, flow_img)
        flow_img = flow_to_image(enc_flow*3, convert_to_bgr=True, div=255)
        tar = os.path.join("/home/nijingcheng/mmaction2/tools/RAFT/visualize", f"{i}_div255.png")
        print(tar)
        cv2.imwrite(tar, flow_img)
        # u = enc_flow[:,:,0]
        # v = enc_flow[:,:,1]
        # for j in range(8):
        #     beta = 0.25*j*np.pi
        #     sin_beta, cos_beta = np.sin(beta), np.cos(beta)
        #     new_u = cos_beta*u - sin_beta*v
        #     new_v = sin_beta*u + cos_beta*v
        #     enc_flow = np.stack((new_u, new_v), axis=-1)
        #     flow_img = flow_to_image(enc_flow, convert_to_bgr=True)
        #     tar = os.path.join("/home/nijingcheng/mmaction2/tools/RAFT/visualize", f"{i}_rtt_{j}.png")
        #     cv2.imwrite(tar, flow_img)
        # flow_img = flow_to_image(enc_flow*(-1), convert_to_bgr=True)
        # tar = os.path.join("/home/nijingcheng/mmaction2/tools/RAFT/visualize", f"{i}_flow_dec3_v2.png")
        # cv2.imwrite(tar, flow_img)

    _ = input()

    # imgs = [get_img_from_nid(anno[0]['nori_id_seq'][1])] * 16
    # # imgs = [get_img_from_nid(nid) for nid in anno[0]['nori_id_seq'][1]]
    # imgs = torch.Tensor(imgs).float()/255   # t, h, w, c
    # print(imgs.shape)
    # imgs = imgs.permute(0, 3, 1, 2) # t, c, h, w
    # imgs = imgs.unsqueeze(0)
    # imgs = flip_transform(imgs).squeeze(0)
    # print(flip_transform._params["batch_prob"])
    # # imgs = torch.flip(imgs, [-1]).squeeze(0)
    # # imgs = tfm(imgs)
    # imgs = imgs.permute(0, 2, 3, 1)*255
    # imgs = imgs.numpy().astype(np.uint8)
    # imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs]
    # print(imgs[0].mean(), imgs[0].max(), imgs[0][:3, :3])

    # for i in range(len(imgs)):
    #     tar = os.path.join("/home/nijingcheng/mmaction2/tools/RAFT/visualize", f"{i+16}_box.png")
    #     cv2.imwrite(tar, imgs[i])