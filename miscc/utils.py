#!/usr/bin/env python

# -*- encoding: utf-8 -*-

"""
@Author  :   Collapsar-G

@License :   (C) Copyright 2021-*

@Contact :   gjf840513468@gmail.com

@File    :   $Visualization.py

@Time    :   $2021.4.8 $16:50

@Desc    :   工具人

"""
import json

import numpy as np
import torch

from miscc.config import cfg


def data2matrix(data, num_user=1, num_item=1):
    data_matrix = np.zeros(shape=(num_user, num_item))
    data_score = np.zeros(shape=(num_user, num_item))
    for u in data.keys():
        for i in data[u].keys():
            data_matrix[u - 1][i - 1] = 1
            data_score[u - 1][i - 1] = data[u][i]
    data_matrix = torch.tensor(data_matrix)
    data_score = torch.tensor(data_score)
    return data_matrix, data_score


def evaluation(test_score, test_matrix, Estimated_R):
    W, H = test_matrix.size()
    denominator = torch.sum(test_matrix)
    # print(denominator,"@@@@@@@@@@@@@@@@@@")
    # print(Estimated_R)
    # print(test_score)
    pre_numerator = torch.mul((test_score - Estimated_R), test_matrix)
    RMSE = torch.sqrt(torch.sum(pre_numerator.pow(2)) / denominator)

    MAE = torch.sum(torch.abs(pre_numerator)) / denominator

    matrix_5 = torch.full((W, H), 0.5)
    if cfg.GPU_ID != "":
        matrix_5 = matrix_5.cuda()

    pre_numerator1 = torch.sign(pre_numerator - matrix_5)
    tmp_test_score = torch.sign(test_score - matrix_5)

    pre_numerator2 = torch.mul((pre_numerator1 == tmp_test_score), test_matrix)
    numerator = torch.sum(pre_numerator2)
    ACC = numerator / denominator

    a = torch.log(Estimated_R)
    b = torch.log(1 - Estimated_R)

    a = torch.where(torch.isinf(a), torch.full_like(a, 0), a)
    b = torch.where(torch.isinf(b), torch.full_like(b, 0), b)

    a = torch.where(torch.isnan(a), torch.full_like(a, 0), a)
    b = torch.where(torch.isnan(b), torch.full_like(b, 0), b)

    tmp_r = test_score
    tmp_r = a * (tmp_r > 0) + b * (tmp_r == 0)
    tmp_r = torch.mul(tmp_r, test_matrix)
    numerator = torch.sum(tmp_r)
    AVG_loglikelihood = numerator / denominator

    return RMSE.cpu().detach().numpy(), MAE.cpu().detach().numpy(), ACC.cpu().detach().numpy(), AVG_loglikelihood.cpu().detach().numpy()


def data_loat_att(type_data, split=""):
    """
    返回从attributes中的数据分析
    :param type_data: "dvd" 或 "video"
    :param split: "train" 或 "test" 或 ""
    :return:
    """
    path = "./DATA/attributes/attributes_%s_sparse" % type_data
    if split != "":
        path += '_%s' % split
    path += ".json"
    with open(path, 'r') as f:
        load_dict = json.load(f)
    return load_dict


def MAE_score(pre_M, M, N):
    MAE = torch.sum(torch.abs(torch.sub(pre_M, M))) / torch.sum(N)
    return MAE


def normalized5(pred):
    max = torch.max(pred)
    min = torch.min(pred)
    pred = (pred/(max-min))*5
    return pred