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
__c.KFM.NUM_EPOCHS = 50000
__c.KFM.DATA_TYPE = "DVD"

