#!/usr/bin/env python

# -*- encoding: utf-8 -*-

"""
@Author  :   Collapsar-G

@License :   (C) Copyright 2021-*

@Contact :   gjf840513468@gmail.com

@File    :   $Visualization.py

@Time    :   $2021.4.8 $16:50

@Desc    :   项目参数

"""
from easydict import EasyDict as edict

__c = edict()
cfg = __c

cfg.GPU_ID = 0
# KFM
__c.KFM = edict()
__c.KFM.K = 300
__c.KFM.LR = 0.001
__c.KFM.NUM_EPOCHS = 10000
__c.KFM.DATA_TYPE = "DVD"
__c.KFM.LR_gamma = 0.99
__c.KFM.lr_step_size = 500
__c.KFM.loss_B = 0.2
__c.KFM.random_seed = 200
__c.KFM.batch_size = 100
__c.KFM.control = "PredictRegularizationConstrainR"  # 3.6899 2.7936
# __c.KFM.control = "Predict_L" # 3.8418 2.9141
# __c.KFM.control = "PredictRegularizationR" # 4.0450 3.0459

__c.batch_size = 128
__c.random_seed = 200

__c.ml_learn = edict()
__c.ml_learn.num_user = 2504
__c.ml_learn.num_item = 2999
__c.ml_learn.num_hidden = 100
__c.ml_learn.num_mask = 5
__c.ml_learn.epochs = 300
__c.ml_learn.type = "dvd"
__c.ml_learn.lr = 0.01
__c.ml_learn.top_n = 10
__c.ml_learn.train_path = './DATA/%s_sparse_train.csv' % __c.ml_learn.type
__c.ml_learn.test_path = './DATA/%s_sparse_test.csv' % __c.ml_learn.type
__c.ml_learn.dropout = 0.1
__c.ml_learn.lam = 0.2
__c.ml_learn.a_fun = "sigmoid"
__c.ml_learn.b_fun = "softmax"
__c.ml_learn.optim = "Adam"
__c.ml_learn.patience = 7
