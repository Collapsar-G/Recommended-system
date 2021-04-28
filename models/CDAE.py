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

import random
import warnings

import numpy as np
import torch
import torch.nn as nn

from miscc.config import cfg

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



