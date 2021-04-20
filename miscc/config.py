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
__c.KFM.loss_B = 0.0
__c.KFM.random_seed = 200
__c.KFM.batch_size = 100
__c.KFM.control = "PredictRegularizationConstrainR" # 0.9692
# __c.KFM.control = "Predict_L" # 3.8418 2.9141
# __c.KFM.control = "PredictRegularizationR" # 4.0450 3.0459
