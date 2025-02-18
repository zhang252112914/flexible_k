from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import numpy as np
import torch
import pandas as pd


def compute_metrics_event(sim_matrix, mask):
    # v2t
    ind_gt = mask
    ind_sort = np.argsort(np.argsort(-sim_matrix)) + 1
    ind_mask = np.ma.array(ind_gt * ind_sort, mask=ind_gt == 0)

    #t2v
    ind_gt_t = ind_gt.T
    ind_sort_t = np.argsort(np.argsort(-sim_matrix.T)) + 1
    ind_mask_t = np.ma.array(ind_gt_t * ind_sort_t, mask=ind_gt_t == 0)

    v2t_text_counts = mask.sum(axis=1)
    bins = [1,2,3,4,5,np.inf] # 视频按照对应文本数量分类为1，2，3，4，5+
    labels = ["1", "2", "3", "4", "5+"]
    groups_v2t = pd.cut(v2t_text_counts, bins, labels=labels, right=False)

    video_indexs = np.argmax(ind_gt_t, axis =1)
    t2v_text_counts = v2t_text_counts[video_indexs]
    groups_t2v = pd.cut(t2v_text_counts, bins, labels=labels, right=False)

    group_metrics_v2t = {}
    group_metrics_t2v = {}
    for group_name in labels:
        v2t_group_idx = np.where(groups_v2t == group_name)[0]
        if len(v2t_group_idx) == 0:
            continue
        group_ind_mask = ind_mask[v2t_group_idx]
        recall_1_v2t = np.mean(np.mean(group_ind_mask <= 1, axis=1))
        recall_5_v2t = np.mean(np.mean(group_ind_mask <= 5, axis=1))
        recall_10_v2t = np.mean(np.mean(group_ind_mask <= 10, axis=1))
        group_metrics_v2t[group_name] = [recall_1_v2t, recall_5_v2t, recall_10_v2t]

        t2v_group_index = np.where(groups_t2v == group_name)[0]
        if len(t2v_group_index) == 0:
            continue
        group_ind_mask_t = ind_mask_t[t2v_group_index]
        recall_1_t2v = np.mean(np.mean(group_ind_mask_t <= 1, axis=1))
        recall_5_t2v = np.mean(np.mean(group_ind_mask_t <= 5, axis=1))
        recall_10_t2v = np.mean(np.mean(group_ind_mask_t <= 10, axis=1))
        group_metrics_t2v[group_name] = [recall_1_t2v, recall_5_t2v, recall_10_t2v]

    return group_metrics_t2v, group_metrics_v2t
