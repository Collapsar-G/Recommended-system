#!/usr/bin/env python

# -*- encoding: utf-8 -*-

"""
@Author  :   Collapsar-G

@License :   (C) Copyright 2021-*

@Contact :   gjf840513468@gmail.com

@File    :   $Visualization.py

@Time    :   $2021.4.10 $16:50

@Desc    :   初始化数据

"""
from miscc.utils import data_loat_att
from miscc.config import cfg

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

def data_load():
    print('-' * 10)
    print("开始加载数据")
    print('-' * 10)
    data_train = data_loat_att(cfg.KFM.DATA_TYPE, "train")
    print("成功加载训练数据")
    print('-' * 10)
    print('-' * 10)
    print("开始构建评分矩阵")
    x, y = int(data_train["userid_max"]) + 1, int(data_train["commodityid_max"]) + 1
    M = np.zeros((x, y))
    N = np.zeros((x, y))
    data_score = data_train["user2commodity"]
    count = 0
    for i in data_score:
        com = data_score[i]
        for j in com:
            score = com[j]
            M[int(i)][int(j)] = score
            N[int(i)][int(j)] = 1
            count += 1
    print("成功构建评分矩阵")
    return M, N, count


def data_loat_test(pred_data):
    data_test = data_loat_att(cfg.KFM.DATA_TYPE, "test")
    print("成功加载测试数据")
    x, y = int(data_test["userid_max"]) + 1, int(data_test["commodityid_max"]) + 1
    pred_M = np.zeros((x, y))
    test_M = np.zeros((x, y))
    N_test = np.zeros((x, y))
    result = []
    data_score_test = data_test["user2commodity"]
    for i in data_score_test:
        com = data_score_test[i]
        for j in com:
            score = com[j]
            pred_M[int(i)][int(j)] = pred_data[int(i)][int(j)]
            test_M[int(i)][int(j)] = score
            N_test[int(i)][int(j)] = 1
            result.append([i, j, pred_data[int(i)][int(j)]])
    return pred_M, test_M, N_test, result


def getdata_ml_learn(train_path, test_path):
    print("#############")
    train_data, test_data = {}, {}
    train_df = pd.read_csv(train_path).iloc[:, :3] - 1
    train_df = train_df.values.tolist()
    for uid, iid, score in train_df:
        train_data.setdefault(uid, {}).setdefault(iid, score)
    test_df = pd.read_csv(test_path).iloc[:, :3] - 1
    test_df = test_df.values.tolist()
    for uid, iid, score in test_df:
        test_data.setdefault(uid, {}).setdefault(iid, score)
    # print(test_data)
    return train_data, test_data


class ml_Dataset(Dataset):
    def __init__(self, data_matrix, data_score):
        self.data_matrix = data_matrix
        self.data_score = data_score

    def __getitem__(self, idx):
        purchase_vec = torch.tensor(self.data_matrix[idx], dtype=torch.float)
        score_vec = torch.tensor(self.data_score[idx], dtype=torch.float)
        uid = torch.tensor([idx, ], dtype=torch.long)
        if cfg.GPU_ID != "":
            purchase_vec = purchase_vec.cuda()
            score_vec = score_vec.cuda()
            uid = uid.cuda()
        return purchase_vec, uid, score_vec

    def __len__(self):
        return len(self.data_score)