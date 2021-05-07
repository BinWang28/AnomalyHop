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
import itertools 

import pickle
from tqdm import tqdm

import numpy as np

from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import mahalanobis
import scipy.spatial.distance as SSD


from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

# my imports
import my_parser
import cw_saab as sb
import mvtec_data_loader as mvtec_loader
from display import plot_fig


def main():

    # - - - - - - - - - - - - - - - -  arguments - - - - - - - - - - - - - - - - 
    args = my_parser.parse_args()
    print("\n######   Arguments:   ######", args)

    KERNEL = [int(item) for item in args.kernel]
    KEEP_COMPONENTS = [int(item) for item in args.num_comp]
    DISTANCE_MEASURE = args.distance_measure
    LAYER_OF_USE = [int(item) for item in args.layer_of_use if int(item) <= len(KERNEL)]
    
    HOP_WEIGHTS = [float(item) for item in args.hop_weights]
    HOP_WEIGHTS = [float(i)/sum(HOP_WEIGHTS) for i in HOP_WEIGHTS]

    CLASS_NAMES = [item for item in args.class_names]

    assert len(LAYER_OF_USE) != 0, "Invalid LAYER_OF_USE" 
    
    total_roc_auc = []
    total_pixel_roc_auc = []
    all_results = {}

    # data loader
    for class_name in CLASS_NAMES:

        # - - - - - - - - - - - - - - - - - - - - Data Loader - - - - - - - - - - - - - - - - - - - - - - - - 

        train_dataset = mvtec_loader.MVTecDataset(args.data_path, class_name=class_name, is_train=True)
        train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True)
        test_dataset = mvtec_loader.MVTecDataset(args.data_path, class_name=class_name, is_train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=32, pin_memory=True)


        # - - - - - - - - - - - - - - - - - - - - Training - - - - - - - - - - - - - - - - - - - - - - - - 

        # extract train set features
        train_feature_filepath = os.path.join(args.save_path, 'train_%s.pkl' % class_name)
        

        print("\n######   Prepare Training Data:   ######")
        all_train_input = []

        for (x, _, _) in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):
            x = x.numpy()
            all_train_input.append(x)

        all_train_input = np.concatenate(all_train_input)

        print("\n######   Saak Training:   ######")

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
        # gather all hops 
        for i_layer in range(len(sb_feature_all)):
            
            # skip unselected layers
            if i_layer+1 not in LAYER_OF_USE:
                train_outputs.append([None, None])
                continue

            train_layer_i_feature = sb_feature_all[i_layer]
            train_layer_i_feature = np.array(train_layer_i_feature)
            B, C, H, W = train_layer_i_feature.shape

            train_layer_i_feature = train_layer_i_feature.reshape(B, C, H * W)
            

            if DISTANCE_MEASURE == 'loc_gaussian':
                # gaussian distance measure            
                mean = np.mean(train_layer_i_feature, 0)
                cov = np.zeros((C, C, H * W))
                conv_inv = np.zeros((C, C, H * W))

                I = np.identity(C)
                for i in range(H * W):
                    cov[:, :, i] = np.cov(train_layer_i_feature[:, :, i], rowvar=False) + 0.01 * I
                    conv_inv[:, :, i] = np.linalg.inv(cov[:, :, i])

                train_outputs.append([mean, conv_inv])

            elif DISTANCE_MEASURE == 'self_ref':
                # pass this process
                pass

            elif DISTANCE_MEASURE == 'glo_gaussian':
                # global gaussian measure
                samples = np.swapaxes(train_layer_i_feature, 1, 2)
                samples = samples.reshape(samples.shape[0]*samples.shape[1], samples.shape[2])
                samples = samples.transpose()
                
                mean = np.mean(samples, 1)
                I = np.identity(C)
                cov = np.cov(samples) + 0.01 * I
                conv_inv = np.linalg.inv(cov)

                train_outputs.append([mean, conv_inv])
                            
        # - - - - - - - - - - - - - - - - - - - - Testing - - - - - - - - - - - - - - - - - - - - - - - - 
        print("\n######   Testing:   ######")

        gt_list = []
        gt_mask_list = []
        test_imgs = []

        for (x, y, mask) in tqdm(test_dataloader, '| feature extraction | test | %s |' % class_name):
            
            test_imgs.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            gt_mask_list.extend(mask.cpu().detach().numpy())
        
        test_imgs = np.stack(test_imgs)

        start_time = time.time()
        s_time = time.time()
        
        _, sb_test_feature_all, _ = sb.inference_chl_wise(sb_params,
                                                            test_imgs, 
                                                            True, 
                                                            -1, 
                                                            len(KERNEL)-1,
                                                            collectFlag=True)

        print('Time for feature extraction: ', time.time() - s_time)

        # show all hops dimensions
        for i in range(len(sb_test_feature_all)):
            print('stage ', i, ': ', sb_test_feature_all[i].shape)

        scores = []
        for i_layer in range(len(sb_test_feature_all)):
            
            # skip unselected layers
            if i_layer+1 not in LAYER_OF_USE:
                train_outputs.append([None, None])
                continue

            test_layer_i_feature = sb_test_feature_all[i_layer]
            test_layer_i_feature = np.array(test_layer_i_feature)

            B, C, H, W = test_layer_i_feature.shape
            test_layer_i_feature = test_layer_i_feature.reshape(B, C, H * W)

            if DISTANCE_MEASURE == 'loc_gaussian':
                # gaussian distance measure           
                dist_list = []
                for i in range(H * W):
                    mean = train_outputs[i_layer][0][:, i]
                    conv_inv = train_outputs[i_layer][1][:, :, i]

                    dist = SSD.cdist(test_layer_i_feature[:,:,i], mean[None, :], metric='mahalanobis', VI=conv_inv)
                    dist = list(itertools.chain(*dist))
                    
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


            elif DISTANCE_MEASURE == 'self_ref':
                # self-reference compute
                dist_list = []
                for sample in test_layer_i_feature:
                    # compute image level mean and covariance
                    mean = np.mean(sample, 1)
                    I = np.identity(C)
                    cov = np.cov(sample) + 0.01 * I
                    conv_inv = np.linalg.inv(cov)

                    dist = SSD.cdist(sample.transpose(), mean[None, :], metric='mahalanobis', VI=conv_inv)
                    dist = list(itertools.chain(*dist))

                    #import pdb; pdb.set_trace()
                    dist = np.array(dist).reshape(H,W)
                    dist_list.append(dist)
                
                dist_list = np.stack(dist_list)

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

            elif DISTANCE_MEASURE == 'glo_gaussian':
                # gaussian distance
                dist_list = []
                
                mean = train_outputs[i_layer][0]
                conv_inv = train_outputs[i_layer][1]

                for i in range(H * W):

                    dist = SSD.cdist(test_layer_i_feature[:,:,i], mean[None, :], metric='mahalanobis', VI=conv_inv)
                    dist = list(itertools.chain(*dist))

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


        # compute final score for all images
        all_scores = []
        all_scores.extend(scores)

        scores_final = np.average(np.stack(all_scores), axis=0, weights=HOP_WEIGHTS)

        end_time = time.time()

        print('Time for testing process: {} for {} images'.format(end_time - start_time,test_imgs.shape[0]))

        # calculate image-level ROC AUC score
        img_scores = scores_final.reshape(scores_final.shape[0], -1).max(axis=1)
        gt_list = np.asarray(gt_list)

        img_roc_auc = roc_auc_score(gt_list, img_scores)
        total_roc_auc.append(img_roc_auc)
        print('image ROCAUC: %.3f' % (img_roc_auc))
        
        # get optimal threshold
        gt_mask = np.asarray(gt_mask_list)
        precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores_final.flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]

        # calculate per-pixel level ROCAUC
        per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores_final.flatten())
        total_pixel_roc_auc.append(per_pixel_rocauc)
        print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))

        save_dir = args.save_path + '/' + f'pictures_' + class_name
        os.makedirs(save_dir, exist_ok=True)
        #plot_fig(test_imgs, scores_final, gt_mask_list, threshold, save_dir, class_name)

        all_results[class_name] = {'image ROCAUC: ': img_roc_auc, 'pixel ROCAUC: ': per_pixel_rocauc}


    #print('Average ROCAUC: %.3f' % np.mean(total_roc_auc))
    #print('Average pixel ROCUAC: %.3f' % np.mean(total_pixel_roc_auc))

    all_results['ALL AVG'] = {'image ROCAUC: ': np.mean(total_roc_auc), 'pixel ROCAUC: ': np.mean(total_pixel_roc_auc)}


    for key, value in all_results.items():
        #print(key, ': ', value)

        # write to record
        with open('./job_logs/'+key+'.txt', 'a') as f:
            f.write('\n\n' + time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime()) + '\n')
            f.write(str(args)+'\n')
            f.write(key + ': ' + str(value))



if __name__ == '__main__':
    main()
