#!/usr/bin/env python

# -*- encoding: utf-8 -*-

"""
@Author  :   Collapsar-G

@License :   (C) Copyright 2021-*

@Contact :   gjf840513468@gmail.com

@File    :   $Visualization.py

@Time    :   $2021.4.7 $21:00

@Desc    :   Data analysis and visualization

"""

import csv
import numpy as np
import json

reader = np.loadtxt('./DATA/dvd.csv', delimiter=",")
users = []
commoditys = []
attributes_users = []
attributes_coms = []
for row in reader:
    users.append(row)

for col in reader:
    commoditys.append(col)

for user in users:
    attributes_user = {"count": 0, "sum": 0, "avg": 0}
    for commodity in user:
        if commodity != 0.:
            attributes_user["count"] += 1
            attributes_user["sum"] += commodity
    if attributes_user["count"] != 0.:
        attributes_user["avg"] = attributes_user["sum"] / attributes_user["count"]
    attributes_users.append(attributes_user)

for commodity in commoditys:
    attributes_commodity = {"count": 0, "sum": 0, "avg": 0}
    for user in commodity:
        if user != 0.:
            attributes_commodity["count"] += 1
            attributes_commodity["sum"] += user
    if attributes_commodity["count"] != 0.:
        attributes_commodity["avg"] = attributes_commodity["sum"] / attributes_commodity["count"]
    attributes_coms.append(attributes_commodity)



attributes = {}
attributes["commoditys"] = attributes_coms
attributes["users"] = attributes_users

string = json.dumps(attributes)
with open(r'./output/attributes_dvd.json', 'w')as f:
    f.write(string)
