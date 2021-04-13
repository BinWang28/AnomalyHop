#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   my_parser.py
@Time    :   2021/03/30 11:40:51
@Author  :   bin.wang
@Version :   1.0
'''

# here put the import lib

import argparse


def parse_args():
    parser = argparse.ArgumentParser('PaDiM')
    parser.add_argument('--data_path', type=str, default='./datasets/mvtec_anomaly_detection')
    parser.add_argument('--save_path', type=str, default='./mvtec_result')
    #parser.add_argument('--arch', type=str, choices=['resnet18', 'wide_resnet50_2'], default='wide_resnet50_2')
    return parser.parse_args()
