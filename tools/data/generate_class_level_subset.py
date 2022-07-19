import argparse
import json
import re
import math
import numpy as np
from mmcv import DictAction
from datamaid2.storage.oss import OSS

import refile
from meghair.utils import io

def load_anno(path):
    with refile.smart_load_from(path) as f:
        annos = io.load(f)
    return annos

def cal_mean_class_acc(cf_mat):
    cls_cnt = cf_mat.sum(axis=1)
    cls_hit = np.diag(cf_mat)

    class_acc = [hit / cnt if cnt else 0.0 for cnt, hit in zip(cls_cnt, cls_hit)]
    mean_class_acc = np.mean(class_acc)
    return mean_class_acc, class_acc

def parse_label(dataset_name):
    if dataset_name == "sthv2":
        oss_path = "s3://activity-public/something-something/somethingv2/something-something-v2-labels.json"
        with refile.smart_load_from(oss_path) as f:
            annos = json.load(f)
        label_map = {int(v):k for k, v in annos.items()}
    else:
        raise ValueError(f"Unrecognized dataset name {dataset_name}.")
    return label_map

def generate_topk_metas(cf_mat, labels, ori_annos, k, is_percentage=False):
    """生成包含Topk类别的标注子集
    Args:
        cf_mat: 混淆矩阵
        labels: 标签id到名称的映射
        ori_annos: 原始标注，字典（可能包含训练和测试标注）
        k: 选择类别数/比例，负数表示选择最低的topk
        is_percentage: 表示k是否是百分比
    """
    _, class_acc = cal_mean_class_acc(cf_mat)
    num_classes = len(class_acc)
    assert num_classes == len(labels), f'{num_classes} vs {len(labels)}'
    top_k = True
    if k < 0:
        top_k = False
        k *= -1
    assert k <= num_classes, f'{k} vs {num_classes}'

    if is_percentage:
        k = math.ceil(k*num_classes)
    acc_list = [(class_acc[i] if top_k else -class_acc[i], i) for i in range(len(class_acc))]
    acc_list.sort(reverse=True)
    choosen_ids = [acc_list[i][1] for i in range(k)]
    print(num_classes, len(labels), choosen_ids)

    results = dict()
    for name, anno in ori_annos.items():
        new_anno = list()
        for item in anno:
            cid = item['label']
            if cid in choosen_ids:
                new_anno.append(item)
        results[name] = new_anno
    
    return results

def parse_args():
    parser = argparse.ArgumentParser(description='extract frames for AVA Dataset')
    parser.add_argument('cf_dir', type=str, help='confusion matrix directory')
    parser.add_argument(
        '--topk',
        type=int,
        default=15,
        help='filter topk classes')
    parser.add_argument(
        '--is-percentage',
        action='store_true',
        help='whether the unit is percentage')
    parser.add_argument(
        '--save',
        action='store_true',
        help='whether to save results')
    parser.add_argument(
        '--dataset-name',
        type=str,
        default='sthv2',
        help='dataset name')
    parser.add_argument(
        '--splits',
        nargs='+',
        action=DictAction,
        default={},
        help='chosen splits, e.g. --splits train=...')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    """
    For use, if function is extract_frame_alpha, then set --level 1 --alpha.
    else, set level = 2
    Example:
    python tools/data/generate_class_level_subset.py ~/mmaction2/work_dirs/clf_train/tsm_r50_1x1x8_40e_sthv2_rgb_triplet_top4_norm/results/cf_latest.pkl
    --topk 10 --save --splits val=s3://activity-public/something-something/annos/somethingv2/somethingv2_val.pkl
    """
    args = parse_args()

    latest_cf = load_anno(args.cf_dir)[1]   # 0->step, 1->cf_mat
    labels = parse_label(args.dataset_name)
    annos = {k: load_anno(v) for k, v in args.splits.items()}
    new_annos = generate_topk_metas(latest_cf, labels, annos, args.topk, args.is_percentage)

    if args.save:
        oss = OSS()
        for split, path in args.splits.items():
            f1 = 'pr' if args.is_percentage else 'rel'
            f2 = 'neg' if args.topk < 0 else 'pos'
            f3 = str(abs(args.topk))
            new_path = re.sub('.pkl', f'_{f1}_{f2}_{f3}.pkl', path)
            new_anno = new_annos[split]
            print(f'New path : {new_path}, Length : {len(new_anno)}')
            oss.put_py_object(new_anno, new_path)