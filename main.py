#!/usr/bin/env python

# -*- encoding: utf-8 -*-

"""
@Author  :   Collapsar-G

@License :   (C) Copyright 2021-*

@Contact :   gjf840513468@gmail.com

@File    :   $Visualization.py

@Time    :   $2021.4.8 $16:50

@Desc    :   项目入口

"""

import argparse

from CDAE_train import train as CDAE
from KFM_train import KFM


def parse_args():
    parser = argparse.ArgumentParser(description='推荐系统')
    parser.add_argument('--type', type=str, help='dvd or video', default="dvd")
    parser.add_argument('--al', type=str, help='algorithm: CDAE or KFM', default="CDAE")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.al == "CDAE":
        CDAE()
    elif args.al == "KFM":
        KFM()
