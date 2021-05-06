#!/usr/bin/env python

# -*- encoding: utf-8 -*-

"""
@Author  :   Collapsar-G

@License :   (C) Copyright 2021-*

@Contact :   gjf840513468@gmail.com

@File    :   $train.py

@Time    :   $2021.4.21 $8：50

@Desc    :   KFM模型训练入口

"""
import os
import time as time

import torch

from dataset import data_load
from miscc.config import cfg
from models.KFM import train

if cfg.GPU_ID != "":
    torch.cuda.set_device(cfg.GPU_ID)


def KFM(control=cfg.KFM.control):
    times = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    path = "./output/KFM/%s" % times
    if not os.path.exists(path):
        os.makedirs(path)

    M, N, count = data_load()

    pred = train(M, N, control, count, path)

    # pred_data = pred.cpu().detach().numpy()
    #
    # np.savetxt("%s/%s_sparse_pred_data.txt" % (path, cfg.KFM.DATA_TYPE), pred_data)
    # print("保存训练结果到:%s/%s_sparse_pred_data_all.txt" % (path, cfg.KFM.DATA_TYPE))
    #
    # print('-' * 10)
    # print('-' * 10)
    #
    # pred_M, test_M, N_test, result = data_loat_test(pred_data)
    #
    # file_write_obj = open("%s/%s_sparse_pred_data_test.txt" % (path, cfg.KFM.DATA_TYPE), 'w')
    # for var in result:
    #     file_write_obj.writelines(var[0] + "," + var[1] + "," + str(var[2]))
    #     # print(var)
    #     file_write_obj.write('\n')
    # file_write_obj.close()
    #
    # # result_data = pd.DataFrame(data=result)
    # # np.savetxt("%s/%s_sparse_pred_data_test.txt" % (path, cfg.KFM.DATA_TYPE), result)
    # # result_data.to_csv("%s/%s_sparse_pred_data_test.txt" % (path, cfg.KFM.DATA_TYPE))
    # print("保存测试结果到:%s/%s_sparse_pred_data_test.txt" % (path, cfg.KFM.DATA_TYPE))
    # test(pred_M, test_M, N_test, path)

    return


if __name__ == "__main__":
    KFM()
