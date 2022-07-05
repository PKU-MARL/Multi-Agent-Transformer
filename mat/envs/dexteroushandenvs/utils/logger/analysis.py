# -*- coding:utf-8 -*-
import csv
import numpy as np

with open('/home/hp-3070/logs/env_speed/plot_figure/ShadowHandCatchAbreastHAPPO/test_rew_4seeds.csv') as csv_file:
    row = csv.reader(csv_file, delimiter='|')  # 分隔符方式

    next(row)  # 读取首行
    leftDataProp = []  # 创建一个数组来存储数据

    # 读取除首行以后每一行的第41列数据，并将其加入到数组leftDataProp之中
    for r in row:
        leftDataProp.append(float(r[0].split(",")[1]))  # 将字符串数据转化为浮点型加入到数组之中

print('方差:', np.var(leftDataProp))   # 输出方差
print('均值:', np.mean(leftDataProp))  # 输出均值