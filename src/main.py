#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2021/03/30 11:40:05
@Author  :   bin.wang
@Version :   1.0
'''

# here put the import lib
import os
import time

import pickle
from tqdm import tqdm

import numpy as np

from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import mahalanobis

from skimage import morphology
from skimage.segmentation import mark_boundaries

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt
import matplotlib


import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

# my imports
import my_parser
import cw_saab as sb
import mvtec_data_loader as mvtec_loader



KERNEL = [5,5,5,5,5]
KEEP_COMPONENTS = [5,5,5,5,5]
MAX_GMM_SAMPLES = 100000
GMM_COMPONENTS = 5
#DISTANCE_MEASURE = ['PIXEL_GAUSS', 'KMEANS', 'GLOBAL_GAUSS'] # 'PIXEL_GAUSS', 'KMEANS', 'GLOBAL_GAUSS'
DISTANCE_MEASURE = ['PIXEL_GAUSS'] # 'PIXEL_GAUSS', 'KMEANS', 'GLOBAL_GAUSS'


def main():

    # arguments
    args = my_parser.parse_args()
    print("\n######   Arguments:   ######\n", args)

    total_roc_auc = []
    total_pixel_roc_auc = []

    all_results = {}

    # data loader
    for class_name in mvtec_loader.CLASS_NAMES:

        train_dataset = mvtec_loader.MVTecDataset(args.data_path, class_name=class_name, is_train=True)
        train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True)
        test_dataset = mvtec_loader.MVTecDataset(args.data_path, class_name=class_name, is_train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=32, pin_memory=True)


        # - - - - - - - - - - - - - - - - - - - - Training - - - - - - - - - - - - - - - - - - - - - - - - 

        # extract train set features
        train_feature_filepath = os.path.join(args.save_path, 'train_%s.pkl' % class_name)
        

        print("\n######   Prepare Training Data:   ######\n")
        all_train_input = []

        for (x, _, _) in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):
            x = x.numpy()
            all_train_input.append(x)

        all_train_input = np.concatenate(all_train_input)

        print("\n######   Saak Training:   ######\n")

        # all_train_input = np.random.rand(209, 3, 224, 224) # okay
        # all_train_input # (209, 3, 224, 224) # no okay


        sb_params, sb_feature_all, sb_feature_last = sb.multi_saab_chl_wise(all_train_input,
                                                                            [1,1,1,1,1], # stride
                                                                            KERNEL, # kernel
                                                                            [1,1,1,1,1], # dilation
                                                                            KEEP_COMPONENTS,
                                                                            0.125,
                                                                            padFlag = [False,False,False,False,False],
                                                                            recFlag = True,
                                                                            collectFlag = True)
        # show all hops dimensions
        for i in range(len(sb_feature_all)):
            print('stage ', i, ': ', sb_feature_all[i].shape)


        train_outputs = []
        k_means_trained = []
        gmm_trained = []

        # gather all hops 
        for i_layer in range(len(sb_feature_all)):

            train_layer_i_feature = sb_feature_all[i_layer]
            train_layer_i_feature = np.array(train_layer_i_feature)
            B, C, H, W = train_layer_i_feature.shape

            train_layer_i_feature = train_layer_i_feature.reshape(B, C, H * W)
            

            if 'PIXEL_GAUSS' in DISTANCE_MEASURE:
                # gaussian distance measure            
                mean = np.mean(train_layer_i_feature, 0)
                cov = np.zeros((C, C, H * W))
                I = np.identity(C)
                for i in range(H * W):
                    cov[:, :, i] = np.cov(train_layer_i_feature[:, :, i], rowvar=False) + 0.01 * I
                
                train_outputs.append([mean, cov])

            '''
            # k-means global distance measure
            print('K-means training for layer: ', i_layer)
            train_layer_i_feature = np.swapaxes(train_layer_i_feature,1,2)
            train_layer_i_feature = train_layer_i_feature.reshape(-1,train_layer_i_feature.shape[2])
            kmeans = KMeans(n_clusters=100, random_state=0)
            kmeans.fit(train_layer_i_feature)
            k_means_trained.append(kmeans)
            '''

            if 'GLOBAL_GAUSS' in DISTANCE_MEASURE:
                # gmm global distance measure
                print('GMM training for layer: ', i_layer+1)
                start_time = time.time()

                train_layer_i_feature = np.swapaxes(train_layer_i_feature,1,2)
                train_layer_i_feature = train_layer_i_feature.reshape(-1,train_layer_i_feature.shape[2])

                # random sample if too much samples, gmm can not handle
                if train_layer_i_feature.shape[0] > MAX_GMM_SAMPLES:
                    print('MAX GMM SAMPLES: ', MAX_GMM_SAMPLES)
                    sample_idx = np.random.randint(train_layer_i_feature.shape[0], size=MAX_GMM_SAMPLES)
                    train_layer_i_feature = train_layer_i_feature[sample_idx,:]

                gmm_model = GaussianMixture(n_components=GMM_COMPONENTS, random_state=0)

                gmm_model.fit(train_layer_i_feature)
                gmm_trained.append(gmm_model)

                end_time = time.time()
                time_elapsed = (end_time - start_time)
                print('Time used: {:.2f} in seconds'.format(time_elapsed))





        # - - - - - - - - - - - - - - - - - - - - Testing - - - - - - - - - - - - - - - - - - - - - - - - 
        gt_list = []
        gt_mask_list = []
        test_imgs = []

        for (x, y, mask) in tqdm(test_dataloader, '| feature extraction | test | %s |' % class_name):
            
            test_imgs.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            gt_mask_list.extend(mask.cpu().detach().numpy())

        test_imgs = np.stack(test_imgs)

        _, sb_test_feature_all, _ = sb.inference_chl_wise(sb_params,
                                                            test_imgs, 
                                                            True, 
                                                            -1, 
                                                            len(KERNEL)-1,
                                                            collectFlag=True)

        # show all hops dimensions
        for i in range(len(sb_test_feature_all)):
            print('stage ', i, ': ', sb_test_feature_all[i].shape)

        scores = []
        k_means_scores = []
        gmm_scores = []

        for i_layer in range(len(sb_test_feature_all)):
            test_layer_i_feature = sb_test_feature_all[i_layer]
            test_layer_i_feature = np.array(test_layer_i_feature)

            B, C, H, W = test_layer_i_feature.shape
            test_layer_i_feature = test_layer_i_feature.reshape(B, C, H * W)
        

            if 'PIXEL_GAUSS' in DISTANCE_MEASURE:
                # gaussian distance measure           
                dist_list = []
                for i in range(H * W):
                    mean = train_outputs[i_layer][0][:, i]
                    conv_inv = np.linalg.inv(train_outputs[i_layer][1][:, :, i])
                    dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in test_layer_i_feature]
                    dist_list.append(dist)

                dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)
            
                # upsample
                dist_list = torch.tensor(dist_list)
                score_map = F.interpolate(dist_list.unsqueeze(1), size=x.size(2), mode='bilinear',
                                        align_corners=False).squeeze().numpy()

                # apply gaussian smoothing on the score map
                for i in range(score_map.shape[0]):
                    score_map[i] = gaussian_filter(score_map[i], sigma=4)
                # Normalization
                max_score = score_map.max()
                min_score = score_map.min()
                score = (score_map - min_score) / (max_score - min_score)
                scores.append(score) # all scores from different hop features

            '''
            # * * * *
            # k-means scoring
            test_layer_i_feature = np.swapaxes(test_layer_i_feature,1,2)
            test_layer_i_feature = test_layer_i_feature.reshape(-1,test_layer_i_feature.shape[2])
            kmeans = k_means_trained[i_layer]
            k_means_distance = kmeans.transform(test_layer_i_feature)
            k_means_distance = np.min(k_means_distance,1)

            k_means_distance = k_means_distance.reshape(B, H, W)
            # upsample
            k_means_distance = torch.tensor(k_means_distance)
            k_mean_score_map = F.interpolate(k_means_distance.unsqueeze(1), size=x.size(2), mode='bilinear',
                                    align_corners=False).squeeze().numpy()
            
            # apply gaussian smoothing on the score map
            for i in range(k_mean_score_map.shape[0]):
                k_mean_score_map[i] = gaussian_filter(k_mean_score_map[i], sigma=4)

            # Normalization
            max_score = k_mean_score_map.max()
            min_score = k_mean_score_map.min()
            score = (k_mean_score_map - min_score) / (max_score - min_score)
            k_means_scores.append(score) # all scores from different hop features
            '''
    
            if 'GLOBAL_GAUSS' in DISTANCE_MEASURE:
                # gmm scoring
                test_layer_i_feature = np.swapaxes(test_layer_i_feature,1,2)
                test_layer_i_feature = test_layer_i_feature.reshape(-1,test_layer_i_feature.shape[2])
                gmm_model = gmm_trained[i_layer]
                gmm_probability = gmm_model.predict_proba(test_layer_i_feature)

                gmm_probability = 1 - np.max(gmm_probability,1)
                gmm_probability = gmm_probability.reshape(B, H, W)
                # upsample
                gmm_probability = torch.tensor(gmm_probability)
                gmm_score_map = F.interpolate(gmm_probability.unsqueeze(1), size=x.size(2), mode='bilinear',
                                        align_corners=False).squeeze().numpy()
                
                # apply gaussian smoothing on the score map
                for i in range(gmm_score_map.shape[0]):
                    gmm_score_map[i] = gaussian_filter(gmm_score_map[i], sigma=4)

                # Normalization
                max_score = gmm_score_map.max()
                min_score = gmm_score_map.min()
                score = (gmm_score_map - min_score) / (max_score - min_score)
                gmm_scores.append(score) # all scores from different hop features


        # compute final score for all images
        all_scores = []
        all_scores.extend(scores)
        all_scores.extend(k_means_scores)
        all_scores.extend(gmm_scores)

        #import pdb; pdb.set_trace()

        #all_scores.extend()
        #scores_final = np.mean(np.stack(scores), 0) # average all hop heat map 
        #scores_final = np.mean(np.stack(k_means_scores), 0) # average all hop heat map
        scores_final = np.mean(np.stack(all_scores), 0) # average all hop heat map


        # calculate image-level ROC AUC score
        img_scores = scores_final.reshape(scores_final.shape[0], -1).max(axis=1)
        gt_list = np.asarray(gt_list)
        #fpr, tpr, _ = roc_curve(gt_list, img_scores)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        total_roc_auc.append(img_roc_auc)
        print('image ROCAUC: %.3f' % (img_roc_auc))
        #fig_img_rocauc.plot(fpr, tpr, label='%s img_ROCAUC: %.3f' % (class_name, img_roc_auc))
        
        # get optimal threshold
        gt_mask = np.asarray(gt_mask_list)
        precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores_final.flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]

        # calculate per-pixel level ROCAUC
        #fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
        per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores_final.flatten())
        total_pixel_roc_auc.append(per_pixel_rocauc)
        print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))

        #fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, per_pixel_rocauc))
        save_dir = args.save_path + '/' + f'pictures_' + class_name
        os.makedirs(save_dir, exist_ok=True)
        plot_fig(test_imgs, scores_final, gt_mask_list, threshold, save_dir, class_name)

        all_results[class_name] = {'image ROCAUC: ': img_roc_auc, 'pixel ROCAUC: ': per_pixel_rocauc}


    print('Average ROCAUC: %.3f' % np.mean(total_roc_auc))
    #fig_img_rocauc.title.set_text('Average image ROCAUC: %.3f' % np.mean(total_roc_auc))
    #fig_img_rocauc.legend(loc="lower right")

    print('Average pixel ROCUAC: %.3f' % np.mean(total_pixel_roc_auc))
    #fig_pixel_rocauc.title.set_text('Average pixel ROCAUC: %.3f' % np.mean(total_pixel_roc_auc))
    #fig_pixel_rocauc.legend(loc="lower right")

    all_results['ALL AVG'] = {'image ROCAUC: ': np.mean(total_roc_auc), 'pixel ROCAUC: ': np.mean(total_pixel_roc_auc)}


    #fig.tight_layout()
    #fig.savefig(os.path.join(args.save_path, 'roc_curve.png'), dpi=100)

    for key, value in all_results.items():
        print(key, ': ', value)


            



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
        ax_img[2].imshow(img, cmap='gray', interpolation='none')
        ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
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

        fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=100)
        plt.close()


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    
    return x



if __name__ == '__main__':
    main()
