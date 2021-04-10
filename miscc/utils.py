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
import torch


def data_loat_att(type_data, split=""):
    """
    返回从attributes中的数据分析
    :param type_data: "dvd" 或 "video"
    :param split: "train" 或 "test" 或 ""
    :return:
    """
    path = "../DATA/attributes/attributes_%s_sparse" % type_data
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