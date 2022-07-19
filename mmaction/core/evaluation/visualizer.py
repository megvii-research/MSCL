import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import pickle
import refile
import json
import warnings

try:
    from .accuracy import confusion_matrix
except:
    from mmaction.core.evaluation import confusion_matrix
from collections import Counter
from megskull.utils.meta import cached_property
from prettytable import PrettyTable

def visualize_cf(cf, sorted_classes, tar_path, eps=1e-7):
    # Normalize cf
    normed_cf = cf/(cf.sum(axis=-1, keepdims=True) + eps)
    normed_cf = np.nan_to_num(normed_cf)
    plt.matshow(normed_cf, cmap=plt.cm.Greens) 
    plt.colorbar()
    for i in range(len(normed_cf)): 
        for j in range(len(normed_cf)):
            plt.annotate('{:.2f}'.format(normed_cf[i,j])[2:], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    plt.xticks(range(len(sorted_classes)), sorted_classes, fontsize='small')
    plt.yticks(range(len(sorted_classes)), sorted_classes, fontsize='small')
    plt.ylabel('True label')
    plt.xlabel('Predicted label') 
    os.makedirs(osp.dirname(tar_path), exist_ok=True)
    plt.savefig(tar_path, dpi=300)

def parse_label(dataset_name):
    if dataset_name == "sthv2":
        oss_path = "s3://activity-public/something-something/somethingv2/something-something-v2-labels.json"
        with refile.smart_load_from(oss_path) as f:
            annos = json.load(f)
        label_map = {int(v):k for k, v in annos.items()}
    else:
        raise ValueError(f"Unrecognized dataset name {dataset_name}.")
    return label_map

def visualize_acc(label_map, acc_list, dir_path, title_info=''):
    os.makedirs(dir_path, exist_ok=True)
    x = PrettyTable(["Cls_name", "Acc", "Acc_cmp"])
    for acc, acc_cmp, idx in acc_list:
        x.add_row([label_map[idx], "{:.2f}".format(acc), "{:.2f}".format(acc_cmp)])
    with open(osp.join(dir_path, "delta_acc.txt"), "w") as f:
        f.write(f"Title: {title_info} \n")
        f.write(str(x))


def save_res(mac, tar_path):
    # Normalize cf
    pickle.dump(mac, open(tar_path, "wb"))

class ClfVisualizer:
    def __init__(self, cur_path, default_path, dataset_name, vis_acc=True, vis_cf=False, k=20) -> None:
        self.k = k
        self.cur_path = cur_path
        self.default_path = default_path
        self.step = 1
        self.label_map = parse_label(dataset_name)

        self.vis_acc = vis_acc
        self.vis_cf = vis_cf
    
    @cached_property
    def sorted_classes(self):
        label_cnt_pairs = list(Counter(self.labels).items())
        label_cnt_pairs.sort(key=lambda x: x[1], reverse=True)  # according to nums
        return [pr[0] for pr in label_cnt_pairs]

    def cal_mean_class_acc(self, cf_mat):
        cls_cnt = cf_mat.sum(axis=1)
        cls_hit = np.diag(cf_mat)

        class_acc = [hit / cnt if cnt else 0.0 for cnt, hit in zip(cls_cnt, cls_hit)]
        mean_class_acc = np.mean(class_acc)
        return mean_class_acc, class_acc

    def visualize(self, scores, labels):
        """Visualization by scores and labels, then return the mean class accuracy.
        // This should not be used by evaluation with simple=True, for this setting
        will use batch results as input !!!
        // 存储的混淆矩阵的类别数取决于预测和GT，如果有一个类没有被预测或存在于GT，
        则会被忽略掉，在比较时可能会产生维度不一致的问题

        Args:
            scores (list[np.ndarray]): Prediction scores for each class.
            labels (list[int]): Ground truth labels.

        Returns:
            np.ndarray: Mean class accuracy.
        """
        pred = np.argmax(scores, axis=1)
        self.labels = labels
        cf_mat = confusion_matrix(pred, labels).astype(float)

        # Cal mean class acc
        mean_class_acc, mac_full = self.cal_mean_class_acc(cf_mat)

        # Save current cf matrix
        # *先存再读，当cur_path和default_path相同时也能看到结果
        save_path = osp.join(self.cur_path, "results")
        os.makedirs(save_path, exist_ok=True)
        with open(osp.join(save_path, "cf_latest.pkl"), "wb") as f:
            pickle.dump((self.step, cf_mat), f)

        # Load cmp cf matrix
        cmp_latest_cf_path = osp.join(self.default_path, "results", "cf_latest.pkl")
        if osp.exists(cmp_latest_cf_path):
            print("Load cmp results: ", cmp_latest_cf_path)
            with open(cmp_latest_cf_path, "rb") as f:
                cmp_step, cmp_cf_mat = pickle.load(f) 
        else:
            warnings.warn("Find no latest file in default_path!")
            return mean_class_acc
        
        # Visualize
        if self.vis_acc:
            _, cmp_mac_full = self.cal_mean_class_acc(cmp_cf_mat)
            abs_acc_list = [(mac_full[i], cmp_mac_full[i], i) for i in range(len(mac_full))]
            abs_acc_list.sort(reverse=True, key=lambda x: x[0]-x[1])

            rel_acc_list = [(mac_full[i], cmp_mac_full[i], i) for i in range(len(mac_full))]
            rel_acc_list.sort(reverse=True, key=lambda x: (x[0]-x[1])/(x[1]+1e-5))

            visualize_acc(self.label_map, abs_acc_list[:self.k], osp.join(save_path, "abs_topk"), f"Step: {cmp_step}")
            visualize_acc(self.label_map, abs_acc_list[-self.k:], osp.join(save_path, "abs_downk"), f"Step: {cmp_step}")
            visualize_acc(self.label_map, rel_acc_list[:self.k], osp.join(save_path, "rel_topk"), f"Step: {cmp_step}")
            visualize_acc(self.label_map, rel_acc_list[-self.k:], osp.join(save_path, "rel_downk"), f"Step: {cmp_step}")
        if self.vis_cf:
            # TopK classes
            topk_chosen_classes = self.sorted_classes[:self.k]
            topk_cf_mat = cf_mat[topk_chosen_classes][:, topk_chosen_classes]  # k,k
            # cal function
            # topk_mean_class_acc, mac_topk = self.cal_mean_class_acc(topk_cf_mat)
            visualize_cf(topk_cf_mat, topk_chosen_classes, osp.join(self.cur_path, "cf_mat", f"topk_{self.step}.jpg"))

            # DownK classes
            downk_chosen_classes = self.sorted_classes[-self.k:]
            downk_cf_mat = cf_mat[downk_chosen_classes][:, downk_chosen_classes]  # k,k
            # cal function
            # downk_mean_class_acc, mac_downk = self.cal_mean_class_acc(downk_cf_mat)
            visualize_cf(downk_cf_mat, downk_chosen_classes, osp.join(self.cur_path, "cf_mat", f"downk_{self.step}.jpg"))
            self.step += 1

        return mean_class_acc


if __name__ == "__main__":
    n = 15
    cvr = ClfVisualizer(k=n)
    scores = np.random.rand(n*n).reshape(n, n)
    labels = np.arange(n)
    cvr.visualize(scores, labels)