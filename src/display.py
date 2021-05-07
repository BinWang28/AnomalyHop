#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   display.py
@Time    :   2021/04/21 19:49:59
@Author  :   bin.wang
@Version :   1.0
'''

# here put the import lib
import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from skimage import morphology
from skimage.segmentation import mark_boundaries

def plot_fig(test_img, scores, gts, threshold, save_dir, class_name):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
        figa = ax_img[2].imshow(img, cmap='gray', interpolation='none')
        figb = ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[2].title.set_text('Predicted heat map')
        ax_img[3].imshow(mask, cmap='gray')
        ax_img[3].title.set_text('Predicted mask')
        ax_img[4].imshow(vis_img)
        ax_img[4].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=1000)

        # image - 1
        plt.imsave(os.path.join(save_dir, class_name + '_image' + '_{}'.format(i)) + '.png', img)
        plt.imsave(os.path.join(save_dir, class_name + '_gt' + '_{}'.format(i)) + '.png', gt, cmap="gray")
        plt.imsave(os.path.join(save_dir, class_name + '_heat_map' + '_{}'.format(i)) + '.png', heat_map, cmap="jet")
        plt.imsave(os.path.join(save_dir, class_name + '_predict_mask' + '_{}'.format(i)) + '.png', mask, cmap="gray")
        plt.imsave(os.path.join(save_dir, class_name + '_predict_mask_with_image' + '_{}'.format(i)) + '.png', vis_img, cmap="gray")

        extent = ax_img[2].get_window_extent().transformed(fig_img.dpi_scale_trans.inverted())
        fig_img.savefig(os.path.join(save_dir, class_name + '_heat_mapk_with_image' + '_{}'.format(i)), bbox_inches=extent, dpi=1000)

        plt.close()
        plt.close()


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    
    return x