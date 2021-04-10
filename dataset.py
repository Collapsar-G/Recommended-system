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
