#!/usr/bin/env python

# -*- encoding: utf-8 -*-

"""
@Author  :   Collapsar-G

@License :   (C) Copyright 2021-*

@Contact :   gjf840513468@gmail.com

@File    :   $models.py

@Time    :   $2021.4.21 $8：50

@Desc    :   模型


"""

import numpy as np

import torch.nn as nn
import os

from dataset import ml_Dataset, getdata_ml_learn
from miscc.config import cfg
from miscc.utils import data2matrix, evaluation

import torch
from torch.utils.data import DataLoader

import datetime
import dateutil.tz
import warnings
import random

# random.seed(cfg.random_seed)

warnings.filterwarnings('ignore')


class CDAE(nn.Module):
    def __init__(self, config):
        self.num_item = config.num_item
        self.num_user = config.num_user
        self.num_hidden = config.num_hidden
        self.drop_rate = config.dropout
        super(CDAE, self).__init__()
        self.item2hidden = nn.Sequential(
            nn.Linear(self.num_item, self.num_hidden),
            nn.Dropout(self.drop_rate)
        )
        self.id2hidden = nn.Embedding(self.num_user, self.num_hidden)
        self.hidden2out = nn.Linear(self.num_hidden, self.num_item)
        if config.a_fun == "sigmoid":
            self.a_fun = nn.Sigmoid()
        elif config.a_fun == "identity":
            self.a_fun = self.identity()
        elif config.a_fun == "tanh":
            self.a_fun = nn.Tanh()
        elif config.a_fun == "relu":
            self.a_fun = nn.ReLU()
        elif config.a_fun == "softmax":
            self.a_fun = nn.Softmax()

        if config.b_fun == "sigmoid":
            self.b_fun = nn.Sigmoid()
        elif config.b_fun == "identity":
            self.b_fun = self.identity()
        elif config.b_fun == "tanh":
            self.b_fun = nn.Tanh()
        elif config.b_fun == "relu":
            self.b_fun = nn.ReLU()
        elif config.b_fun == "softmax":
            self.b_fun = nn.Softmax()

        self.connect = nn.Linear(self.num_item, self.num_item)

    def regularization(self, lam):
        regul = torch.tensor(0.).cuda()
        temp = torch.tensor(0.).cuda()
        for name, parameters in self.named_parameters():
            temp = torch.norm(parameters).pow(2)
        regul += temp
        regul = regul / 2 * lam
        return regul

    def get_corruption_mask(self, config, uid, purchase_vec):
        batch_size, num_item = purchase_vec.size()
        if config.num_mask != 0:

            num = torch.sum(purchase_vec).cpu().int().numpy()
            # print(num, "$$$$$$$$$$")

            # idx = torch.full((1, batch_size * num_item), 1, dtype=torch.float)
            # idx = idx.view(-1, batch_size * num_item)
            # # print(idx)
            # num_mask = min(int(config.num_mask * num), batch_size * num_item, 2 ** 24 - 1)
            # # print(num_mask)
            # # print(int(config.num_mask * num), batch_size * num_item, 2 ** 24)
            #
            # tmp_zr = torch.multinomial(idx, num_mask, replacement=False)
            # idx = torch.full((1, batch_size * num_item), 0, dtype=torch.float)
            # idx = idx.view(-1, batch_size * num_item)
            # # print(idx)
            # idx[0][tmp_zr] = 1.

            idx = list(range(0, batch_size * num_item))
            tmp_zr = random.sample(idx, int(config.num_mask * num))
            # print(tmp_zr)
            idx = np.zeros(batch_size * num_item, dtype=float)
            for i in tmp_zr:
                idx[i] = 1
            idx = torch.from_numpy(idx)
            if cfg.GPU_ID != "":
                idx = idx.cuda()
            idx = idx.view((batch_size, num_item))
            return idx
        else:
            idx = torch.full((batch_size, num_item), 0, dtype=torch.float)
            if cfg.GPU_ID != "":
                idx = idx.cuda()
            return idx

    def loss(self, out, corrupted_purchase, corrupted_score, config, score_vec):

        # out = out * 4 + 1
        batch_size, _ = corrupted_purchase.size()
        num = torch.sum(corrupted_purchase)
        # cost1 = torch.sum(torch.abs(torch.mul((out - corrupted_score), corrupted_purchase))) / batch_size
        cost1 = torch.sum(torch.abs(torch.mul((out - corrupted_score), score_vec))) / batch_size
        cost2 = self.regularization(config.lam)
        return cost2 + cost1

    def forward(self, config, uid, purchase_vec, score_vec, split="train"):
        # print(uid,purchase_vec)
        if split == "train":

            idx = self.get_corruption_mask(config, uid, purchase_vec)
            # print(idx, "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            mask_vec = torch.sign(idx + purchase_vec)
            corrupted_score = torch.mul(mask_vec, score_vec).float()
            corrupted_purchase = mask_vec
        else:
            corrupted_score = score_vec
            corrupted_purchase = purchase_vec
        # print(corrupted_score.shape)
        # print(uid.type, "@@@@@@@@@")
        a = self.item2hidden(corrupted_score)
        hidden = self.a_fun(self.id2hidden(uid).squeeze(dim=1) + a)
        out_hidden = self.hidden2out(hidden)

        out_hidden = self.b_fun(out_hidden)
        out = self.connect(out_hidden)
        cost = self.loss(out, corrupted_purchase, corrupted_score, config, score_vec)
        return out, cost


def test(model, test_dataloader, var_dataload, config):
    model.eval()
    for purchase_vec, uid, score_vec in var_dataload:
        out, loss = model(config, uid, purchase_vec, score_vec, split="test")
        # print(out)
        for test_purchase_vec, test_uid, test_score_vec in test_dataloader:
            RMSE, MAE, ACC, AVG_loglikelihood = evaluation(test_score_vec, test_purchase_vec, out)
            return RMSE, MAE, ACC, AVG_loglikelihood


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)


def train():
    starttime = datetime.datetime.now()
    # 设置随机数种子
    setup_seed(cfg.random_seed)

    dataset = "ml_learn"

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/CDAE/%s_%s' % \
                 (dataset, timestamp)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_dir + "/terminal.txt", 'w') as f:
        f.write('dataset:%s \n' % dataset)
    config = cfg[dataset]
    with open(output_dir + "/terminal.txt", 'a') as f:
        f.write('config:%s \n' % config)

    if cfg.GPU_ID != "":
        torch.cuda.set_device(cfg.GPU_ID)

    print("数据集：%s" % dataset)
    train_data, test_data = {}, {}

    train_data, test_data = getdata_ml_learn(config.train_path, config.test_path)
    train_matrix, train_score = data2matrix(train_data, num_user=config.num_user, num_item=config.num_item)
    test_matrix, test_score = data2matrix(test_data, num_user=config.num_user, num_item=config.num_item)
    if cfg.GPU_ID != "":
        train_matrix = train_matrix.cuda()
        test_matrix = test_matrix.cuda()
        train_score = train_score.cuda()
        test_score = test_score.cuda()
    print("成功构建评分矩阵")
    print("=" * 15)

    train_dataset = ml_Dataset(train_matrix, train_score)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    test_dataset = ml_Dataset(test_matrix, test_score)
    test_dataload = DataLoader(dataset=test_dataset, batch_size=config.num_user, shuffle=True, drop_last=False)
    var_dataload = DataLoader(dataset=train_dataset, batch_size=config.num_user, shuffle=True, drop_last=False)
    print("DataLoader加载完成")

    model = CDAE(config)
    if cfg.GPU_ID != "":
        model.cuda()
    print("模型构建完成")
    opt = torch.optim
    if config.optim == "Adam":
        opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    elif config.optim == "SGD":
        opt = torch.optim.SGD(model.parameters(), lr=config.lr)
    # model_gard = [1]
    #
    # lr_lambda = lambda epoch: 1-0. * model_gard[epoch]/sum(model_gard)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.8,
                                                           patience=config.patience * config.num_user / cfg.batch_size,
                                                           verbose=False, threshold=0.0001, threshold_mode='rel',
                                                           cooldown=0,
                                                           min_lr=0, eps=1e-08)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=[0.95])
    epoche_list = []
    precision_list = []
    total_loss_list = []
    RMSE, MAE, ACC, AVG_loglikelihood = test(model, test_dataload, var_dataload,config)
    print("RMSE = {:.4f}".format(RMSE), "MAE = {:.4f}".format(MAE), "ACC = {:.10f}".format(ACC),
          "AVG Loglike = {:.4f}".format(AVG_loglikelihood))
    print('-' * 55)
    for epoch in range(config.epochs):
        model.train()
        total_loss = torch.tensor(0.).cuda()
        for purchase_vec, uid, score_vec in train_dataloader:
            out, loss = model(config, uid, purchase_vec, score_vec, split="train")

            # print(loss)
            total_loss += loss
            opt.zero_grad()
            loss.backward()
            # model_gard.append([x.grad for x in opt.param_groups[0]['params']])
            # print([x.grad for x in opt.param_groups[0]['params']], "$$$$$$$$$$$$$")
            opt.step()
            scheduler.step(1)
        total_loss_list.append((epoch, total_loss, opt.state_dict()['param_groups'][0]['lr']))
        str = '| epoch {:3d} /{:5d}  | loss {:5.7f} | learning rate {:5.7f}'.format(epoch, config.epochs, total_loss,
                                                                                    opt.state_dict()['param_groups'][0][
                                                                                        'lr'])

        print(str)

        RMSE, MAE, ACC, AVG_loglikelihood = test(model, test_dataload, var_dataload, config)
        print("|RMSE = {:.4f}|".format(RMSE), "MAE = {:.4f}|".format(MAE), "ACC = {:.10f}|".format(ACC),
              "AVG Loglike = {:.4f}|".format(AVG_loglikelihood))
        print('-' * 55)

        if (epoch % 10 == 0) & (epoch != 0):
            with open(output_dir + "/terminal.txt", 'a') as f:
                f.write(str + ' \n')
                f.write('-' * 55 + ' \n')
            print('*' * 55)
            torch.save(model.state_dict(), output_dir + "/models_%d.pth" % epoch)
            print("模型已存储至:%s" % (output_dir + "/models_%d.pth" % epoch))
            print('-' * 55)

    endtime = datetime.datetime.now()
    time_all = (endtime - starttime).seconds
    print("训练完成，用时：", time_all)


if __name__ == "__main__":
    train()
