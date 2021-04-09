#!/usr/bin/env python

# -*- encoding: utf-8 -*-

"""
@Author  :   Collapsar-G

@License :   (C) Copyright 2021-*

@Contact :   gjf840513468@gmail.com

@File    :   $Visualization.py

@Time    :   $2021.4.8 $16:50

@Desc    :   用KFN算法来做矩阵分解

"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import date, datetime
import logging
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable

from miscc.config import cfg
from miscc.utils import data_loat_att

if cfg.GPU_ID != "":
    torch.cuda.set_device(cfg.GPU_ID)

"""
三种不同的loss
"""


def Predict_L(P, Q, M, N):
    """

    :param N: 0/1矩阵
    :param P: 分解矩阵
    :param Q: 分解矩阵
    :param M: 原始稀疏矩阵
    :return: loss：损失值
    """
    predict_M = torch.mm(P, Q.t())
    m, n = M.shape
    loss = 0
    # for i in range(m):
    #     for j in range(n):
    #         if M[i][j] > 0:
    #             loss = loss + (predict_M[i][j] - M[i][j]) ** 2
    loss = torch.sum(torch.abs(torch.pow(predict_M - M, 2)))
    # loss = torch.sum(torch.abs(predict_M.mul(N) - M, )) / torch.sum(N)
    # print(torch.abs(torch.pow(predict_M.mul(N) - M, 2)))
    # print(torch.sum(N   ))
    return loss


def PredictRegularizationR(P, Q, M, N):
    """
    FunkSVD+Regularization
    """
    B = 0.02  # 正则化的系数
    predict_M = torch.mm(P, Q.t())  # 矩阵相乘
    loss = torch.sum(torch.pow(predict_M - M, 2)) + B * torch.sum(torch.pow(P, 2)) + torch.sum(torch.pow(Q, 2))
    # loss = torch.sum(torch.abs(predict_M.mul(N) - M, )) + B * torch.sum(torch.pow(P, 2)) + torch.sum(torch.pow(Q, 2))
    return loss


def PredictRegularizationConstrainR(P, Q, M, N):
    """
    FunkSVD+Regularization+矩阵R的约束(取值只能是0-5, P,Q>0)
    """
    B = 0.01  # 正则化的系数
    x, y = M.shape
    mean_M = torch.mean(torch.mean(M)).float()
    # print(mean_M)
    M_mean = torch.full((x, y), mean_M).cuda()
    predict_M = torch.mm(P, Q.t()) + M_mean  # 矩阵相乘
    # predict_M = torch.mm(P, Q.t())
    # loss = torch.sum(torch.abs(predict_M.mul(N) - M, )) + B * torch.sum(torch.abs(P)) + torch.sum(torch.abs(Q))
    loss = torch.sum(torch.pow(predict_M - M, 2)) + B * torch.sum(torch.pow(P, 2)) + torch.sum(torch.pow(Q, 2))

    N_1 = torch.ones((x, y)).cuda() - N
    N_2_5 = torch.full((x, y), 2.5).cuda()
    # 限定M的范围
    # constraint = torch.sum(N_1.mul(torch.sum(torch.abs(predict_M - N_2_5) - N_2_5)))
    # loss += constraint
    # # 限定P、Q的范围
    # loss += (torch.abs(P[P < 0]).sum() + (torch.abs(Q[Q < 0])).sum())

    constraint = torch.sum(N_1.mul(torch.pow(torch.abs(predict_M - N_2_5) - N_2_5, 2)))
    loss += constraint
    # 限定P、Q的范围
    loss += ((P[P < 0] ** 2).sum() + (Q[Q < 0] ** 2).sum())
    return loss


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train(M, N, control, count):
    M = M
    n, m = M.shape
    K = cfg.KFM.K
    M = torch.from_numpy(M).float().cuda()
    N = torch.from_numpy(N).float().cuda()

    # 初始化矩阵P和Q
    P = Variable(torch.randn(n, K)).cuda()
    Q = Variable(torch.randn(m, K)).cuda()
    P.requires_grad = True
    Q.requires_grad = True
    # 定义优化器
    learning_rate = cfg.KFM.LR
    optimizer = torch.optim.Adam([P, Q], lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=1)

    num_epochs = cfg.KFM.NUM_EPOCHS
    loss = 0.
    for epoch in range(num_epochs):

        # 计算Loss
        if control == "Predict_L":
            loss = Predict_L(P, Q, M, N)
        elif control == "PredictRegularizationR":
            loss = PredictRegularizationR(P, Q, M, N)
        elif control == "PredictRegularizationConstrainR":
            loss = PredictRegularizationConstrainR(P, Q, M, N)

            # 反向传播, 优化矩阵P和Q
        optimizer.zero_grad()  # 优化器梯度都要清0
        loss.backward()  # 反向传播
        optimizer.step()  # 进行优化
        lr_scheduler.step()
        if epoch % 100 == 0:
            print('| epoch {:3d} /{:5d}  | '
                  'loss {:5.7f} | '
                  'learning rate {:5.7f}'.format(epoch, num_epochs, loss / count,
                                                 optimizer.state_dict()['param_groups'][0]['lr']))
            loss = 0
    # 求出最终的矩阵P和Q, 与P*Q
    pred = torch.mm(P, Q.t())
    pred = torch.sigmoid(pred) * 5
    return pred


def KMF(control="Predict_L"):
    print('-' * 10)
    print("加载数据")
    print('-' * 10)
    data_train = data_loat_att(cfg.KFM.DATA_TYPE, "train")
    print("成功加载训练数据")
    data_test = data_loat_att(cfg.KFM.DATA_TYPE, "test")
    print("成功加载测试数据")
    print('-' * 10)
    print('-' * 10)

    print("构建评分矩阵")
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
    print('-' * 10)
    print('-' * 10)
    print("开始训练")
    # 设置随机数种子
    setup_seed(200)
    pred = train(M, N, control, count)
    print("训练完成")
    print('-' * 10)
    print('-' * 10)
    print("开始测试")
    pred_data = pred.cpu().detach().numpy()
    np.savetxt("filename.txt", pred_data)
    MAE_list = []
    list = []
    data_score_test = data_test["user2commodity"]
    count = 0
    for i in data_score_test:
        com = data_score_test[i]
        for j in com:
            list.append((i, j))
            MAE_list.append(abs(com[j] - pred_data[int(i)][int(j)]))
    MAE = np.mean(MAE_list)
    print("测试完成")
    print("MAE:", MAE)
    print(list)
    print(MAE_list)
    return


if __name__ == "__main__":
    KMF("PredictRegularizationR")
