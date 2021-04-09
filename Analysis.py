#!/usr/bin/env python

# -*- encoding: utf-8 -*-

"""
@Author  :   Collapsar-G

@License :   (C) Copyright 2021-*

@Contact :   gjf840513468@gmail.com

@File    :   $Visualization.py

@Time    :   $2021.4.7 $21:00

@Desc    :   Data analysis

"""

import csv
import numpy as np
import json


def analysis(path):
    """
    分析训练集的相关信息
    :param path:数据文件路径
    :return attributes:数据的相关属性，包括用户id、商品id、用户评分属性（数量、方差、均值、标准差）、商品评分属性、用户评分列表、商品评分列表
    """
    datapath = "./DATA/%s.csv" % path
    print("处理%s" % datapath)
    reader = np.loadtxt(datapath, delimiter=",")
    users = set()  # 用户id
    commoditys = set()  # 商品id
    attributes_users = {}  # 用户属性
    attributes_coms = {}  # 商品属性
    reuser = {}  # 用户评分列表
    recomm = {}  # 商品评分列表

    user2com2score = {}
    com2user2score = {}
    for row in reader:
        users.add(str(int(row[0])))

        commoditys.add(str(int(row[1])))
    users = list(users)
    commoditys = list(commoditys)

    for user in users:
        reuser[user] = []
        user2com2score[user] = {}
    for comm in commoditys:
        recomm[comm] = []
        com2user2score[comm] = {}

    for row in reader:
        reuser[str(int(row[0]))].append(row[2])
        user2com2score[str(int(row[0]))][str(int(row[1]))] = row[2]
        recomm[str(int(row[1]))].append(row[2])
        com2user2score[str(int(row[1]))][str(int(row[0]))] = row[2]

    for key in reuser:
        value = reuser[key]
        attributes_user = {"count": len(value), "sum": sum(value), "avg": np.mean(value), "var": np.var(value),
                           "std": np.std(value)}

        attributes_users[key] = attributes_user

    for key in recomm:
        value = recomm[key]
        attributes_commodity = {"count": len(value), "sum": sum(value), "avg": np.mean(value), "var": np.var(value),
                                "std": np.std(value)}

        attributes_coms[key] = attributes_commodity

    attributes = {"sum_user": len(users), "sum_commodity": len(commoditys), "userid_max": max([int(item) for item in users]),
                  "commodityid_max": max([int(item) for item in commoditys]), "attributes_commoditys": attributes_coms,
                  "attributes_users": attributes_users, "users": users, "commoditys": commoditys,
                  "user2commodity": user2com2score, "commodity2user": com2user2score}
    return attributes


def save_attributes(attributes, path):
    """
    将attributes字典保存到json文件中。
    :param attributes: 数据属性字典
    :param path: 保存文件名
    :return:
    """
    string = json.dumps(attributes)
    outpath = r'./output/attributes/attributes_%s.json' % path
    with open(outpath, 'w')as f:
        f.write(string)


if __name__ == "__main__":
    data_path = ['dvd_sparse_train', 'video_sparse_train', "dvd_sparse_test", 'video_sparse_test', 'dvd_sparse',
                 'video_sparse']

    for path in data_path:
        save_attributes(analysis(path), path)
