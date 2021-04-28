#!/usr/bin/env python

# -*- encoding: utf-8 -*-

"""
@Author  :   Collapsar-G

@License :   (C) Copyright 2021-*

@Contact :   gjf840513468@gmail.com

@File    :   $KFM.py

@Time    :   $2021.4.8 $16:50

@Desc    :   用KFN算法来做矩阵分解

:TODO:
    1. 将约束修改到1-5;
    2. 训练完成后归一化到1-5范围内;

"""

import numpy as np
import torch
from torch.autograd import Variable

from miscc.config import cfg
from miscc.utils import MAE_score
from miscc.utils import normalized5

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
    loss = torch.sum(torch.abs(torch.pow(torch.sub(predict_M, M.mul(N)), 2)))
    # loss = torch.sum(torch.abs(torch.sub(predict_M, M)))
    return loss


def PredictRegularizationR(P, Q, M, N):
    """
    FunkSVD+Regularization
    """
    B = cfg.KFM.loss_B  # 正则化的系数
    predict_M = torch.mm(P, Q.t())
    # loss = torch.sum(torch.pow(torch.sub(predict_M, M.mul(N)), 2)) + B * torch.sum(torch.pow(P, 2)) + torch.sum(
    # torch.pow(Q, 2))
    loss = torch.sum(torch.abs(torch.sub(predict_M, M))) + B * torch.sum(torch.pow(P, 2)) + torch.sum(
        torch.pow(Q, 2))
    return loss


def PredictRegularizationConstrainR(P, Q, M, N):
    """
    FunkSVD+Regularization+矩阵R的约束(取值只能是1-5, P,Q>0)
    """
    B = cfg.KFM.loss_B  # 正则化的系数
    predict_M = torch.mm(P, Q.t())
    loss = torch.sum(torch.pow(torch.sub(predict_M, M.mul(N)), 2)) + B * torch.sum(torch.pow(P, 2)) + torch.sum(
        torch.pow(Q, 2))
    x, y = M.shape
    N_1 = torch.sub(torch.ones((x, y)).cuda(), N)
    N_3 = torch.full((x, y), 3).cuda()
    N_2 = torch.full((x, y), 2).cuda()
    # 限定M的范围
    constraint = torch.sum(N_1.mul(torch.pow(torch.sub(torch.abs(torch.sub(predict_M, N_3)), N_2), 2)))
    loss += constraint
    # 限定P、Q的范围
    loss += torch.sum(torch.pow(torch.sub(torch.abs(P), P).mul(torch.full(P.shape, 0.5).cuda()), 2)) + torch.sum(
        torch.pow(torch.sub(torch.abs(Q), Q).mul(torch.full(Q.shape, 0.5).cuda()), 2))
    return loss


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train(M, N, control, count):
    print("开始训练")
    # 设置随机数种子
    setup_seed(cfg.KFM.random_seed)
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
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.KFM.lr_step_size, gamma=cfg.KFM.LR_gamma)

    num_epochs = cfg.KFM.NUM_EPOCHS
    loss = 0.
    for epoch in range(num_epochs + 1):

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
        if epoch % cfg.KFM.batch_size == 0:
            print('| epoch {:3d} /{:5d}  | '
                  'loss {:5.7f} | '
                  'learning rate {:5.7f}'.format(epoch, num_epochs, loss / count,
                                                 optimizer.state_dict()['param_groups'][0]['lr']))
            loss = 0

    # 求出最终的矩阵P和Q, 与P*Q
    pred = torch.mm(P, Q.t())
    # pred = torch.sigmoid(pred) * 5
    pred = normalized5(pred)
    print("训练完成")

    print('-' * 10)
    print('-' * 10)
    return pred


def test(pre_M, test_M, N, path):
    print("开始测试")
    pre_M = torch.from_numpy(pre_M).float().cuda()
    test_M = torch.from_numpy(test_M).float().cuda()
    N = torch.from_numpy(N).float().cuda()
    MAE = MAE_score(pre_M.mul(N), test_M, N)
    # MAE = torch.sum(torch.abs(torch.sub(pre_M, test_M))) / torch.sum(N)
    print("MAE:", MAE)
    with open(path + "/terminal.txt", 'w') as f:
        f.write('MAE:%s \n' % MAE)
    print("测试完成")
    return
