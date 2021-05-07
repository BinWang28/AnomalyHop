#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   my_parser.py
@Time    :   2021/03/30 11:40:51
@Author  :   bin.wang
@Version :   1.0
'''

import argparse

def parse_args():
    
    parser = argparse.ArgumentParser('AnomalyHop')

    parser.add_argument('--data_path', type=str, default='./datasets')
    parser.add_argument('--save_path', type=str, default='./mvtec_result')
    parser.add_argument('--kernel', nargs='+', help='kernel sizes as a list')
    parser.add_argument('--num_comp', nargs='+', help='number of components kept for each stage')
    parser.add_argument('--distance_measure', type=str, choices=['self_ref','loc_gaussian', 'glo_gaussian'])
    parser.add_argument('--layer_of_use', nargs='+', help='layers output used to compute gaussian')
    parser.add_argument('--hop_weights', nargs='+', help='weights for each hop')
    parser.add_argument('--class_names', nargs='+', help='classes for evaluation')

    return parser.parse_args()

