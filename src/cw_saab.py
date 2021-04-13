#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 17:15:43 2020
@author: Wei Wang
Revised on 05/23/20, Wei Wang
Revised on 05/30/20, Bin Wang for anomaly detection
"""



import numpy as np
from numpy import linalg as LA
from skimage.util.shape import view_as_windows
from skimage.measure import block_reduce
from itertools import product

from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error as MSE

import gc

#%%
def window_process4(samples, kernel_size, stride, dilate, padFlag=True):
    '''
    @ Args:
        samples(np.array) [num, c, h, w]
    @ Returns:
        patches(np.array) [n, h, w, c]
    '''
    if padFlag:
        samples2 = np.pad(samples,((0,0),(0,0),(int(kernel_size/2),int(kernel_size/2)),(int(kernel_size/2),int(kernel_size/2))),'reflect')
    else:
        samples2 = samples
#    print('-- window_process4 cuboid after patching', samples2.shape) 
    n, c, h, w= samples2.shape
    output_h = (h - kernel_size) // stride + 1
    output_w = (w - kernel_size) // stride + 1
    patches = view_as_windows(np.ascontiguousarray(samples2), (1, c, kernel_size, kernel_size), step=(1, c, stride, stride))
#    patches = view_as_windows(np.ascontiguousarray(samples2), (2, c//2, kernel_size, kernel_size), step=(2, c//2, stride, stride))
    #bin, print('-- window_process4 cuboid after view_as_windows', patches.shape) 
    # --> [output_n=n, output_c==1, output_h, output_w, 4d_kernel_n==1, 4d_kernel_c==c, 4d_kernel_h, 4d_kernel_w]
    patches = patches.reshape(n,output_h, output_w, c, kernel_size, kernel_size)
    assert dilate >=1
    patches = patches[:,:,:,:,::dilate,::dilate]   # arbitary dilate, not necessary to take 9 positions
    patches = patches.reshape(n,output_h,output_w,-1)   # [n,output_h,output_w,c*9]
    return patches  # [n,output_h,output_w,c*9]




def remove_mean(features, axis):
    '''
    Remove the dataset mean.
    :param features [num_samples,...]
    :param axis the axis to compute mean
    '''
    feature_mean = np.mean(features, axis=axis, keepdims=True)
    feature_remove_mean = features - feature_mean
    return feature_remove_mean, feature_mean


def find_kernels_pca(samples, num_kernel, rank_ac, recFlag):
    '''
    Train PCA based on the provided samples.
    If num_kernel can be int (w.r.t. absolute channel number) or float (w.r.t. preserved energy) type.
    If want to keep all energy, feed string 'full' for num_kernel. 
    @Args:
        samples(numpy.array, numpy.float64), [num_samples, num_feature_dimension]: 2D AC sample matrix for PCA
        num_kernel(int, float, or string 'full'): num of channels (or energy percentage) to be preserved
    @Returns:
        pca(sklearn.decomposition.PCA object): trained PCA object
        num_components(int): number of valid AC components, considering constrains of num_kernel(int or float by energy) and rank_ac (if recFlag==True)
        intermediate_ind(1d np.array): conditional energy of valid AC components
    '''
    #bin, print('find_kernels_pca/samples.shape:', samples.shape)
    # determination by number of kernels
    if num_kernel == 'full': #or num_kernel > min(samples.shape):
        pca = PCA(svd_solver = 'full')
    # determination by energy threshold
    else:
        pca = PCA(n_components = num_kernel, svd_solver = 'full')
        #pca = IncrementalPCA(n_components = num_kernel, batch_size=100)
    pca.fit(samples)
    
    num_components = len(pca.components_)
    # consider matrix rank, control orthonormal basis
    #if recFlag: 
    #    num_components = min(pca.n_components_, rank_ac)
    
#    energy_perc = pca.explained_variance_ratio_[:num_components]
#    intermediate_ind = energy_perc > ene_thre
#    print('energy_perc', energy_perc)
#    eneP = np.cumsum(pca.explained_variance_ratio_[:num_components])[-1]
    intermediate_ind = pca.explained_variance_ratio_[:num_components]
    return pca, num_components, intermediate_ind


def conv_with_bias(layer_idx, channel_idx, 
                   sample_init,
                   bias_pre, param_dict, 
                   train_pack=None, train_flag=True):
    '''
    work for each intermediate node (one channel)
    determine intermediate and leaf at each stage when training, set indicator in intermediate_ind_channel to be zero if < energy_thre
    @ Args:
        sample_init: [n,1,h,w]
    @ Returns:
        sample_channel: [n, c, h_new, w_new]
    '''
    if train_flag:
        energy_parent = train_pack
    
    # add bias
    sample = sample_init + bias_pre
    
    sample_patches = window_process4(sample, 
                                     param_dict['kernel_size'][layer_idx], 
                                     param_dict['stride'][layer_idx], 
                                     param_dict['dilate'][layer_idx], 
                                     padFlag=param_dict['padFlag'][layer_idx])  # collect neighbor info
    # --> [n,h,w,c]
    [output_n, output_h, output_w, conved_c] = sample_patches.shape
#    print('- conv_with_bias Sample/cuboid_current.shape:', sample_patches.shape)
    
    # Flatten
    sample_patches = sample_patches.reshape([-1, sample_patches.shape[-1]])
#    sample_patches = sample_patches[interest_sample_idx]
#    print('sample norm', LA.norm(sample_patches, axis=1))
    #bin, print('- conv_with_bias Sample/flatten.shape:', sample_patches.shape)
    
    # get dc response
    sample_patches_ac, dc = remove_mean(sample_patches, axis=1)  # Remove patch mean
    del sample_patches
    gc.collect()
    
    # get ac response (& filters)
    ## Remove feature mean (Set E(X)=0 for each dimension)
    if train_flag:
        sample_patches_centered, feature_expectation = remove_mean(sample_patches_ac, axis=0)
        param_dict['Layer_%d/Slice_%d/feature_expectation' % (layer_idx, channel_idx)] = feature_expectation
    else:
        feature_expectation = param_dict['Layer_%d/Slice_%d/feature_expectation' % (layer_idx, channel_idx)]
        sample_patches_centered = sample_patches_ac - feature_expectation
    del sample_patches_ac
    gc.collect()
    
    # Compute PCA kernel for AC components
    if train_flag:
        rank_ac = np.linalg.matrix_rank(sample_patches_centered)
        ac_pca, num_acKernel, intermediate_ind_channel = find_kernels_pca(sample_patches_centered, param_dict['num_kernels'][layer_idx], rank_ac, param_dict['recFlag'])
        param_dict['Layer_%d/Slice_%d/rank_ac' % (layer_idx, channel_idx)] = rank_ac
        param_dict['Layer_%d/Slice_%d/num_acKernel' % (layer_idx, channel_idx)] = num_acKernel
        param_dict['Layer_%d/Slice_%d/ac_kernels' % (layer_idx, channel_idx)] = ac_pca.components_[:num_acKernel]
        
        # adjust intermediate_ind_channel for dc component, that keep dc component as one leaf node
        intermediate_ind_channel = intermediate_ind_channel.tolist()
        intermediate_ind_channel.insert(0, 1)
#        print('intermediate_ind_channel', intermediate_ind_channel)
        intermediate_ind_channel = np.array(intermediate_ind_channel) * energy_parent
#        intermediate_ind_channel[intermediate_ind_channel < param_dict['energy_perc_thre']] = 0
#        print('intermediate_ind_channel', intermediate_ind_channel)
        param_dict['Layer_%d/Slice_%d/intermediate_ind' % (layer_idx, channel_idx)] = intermediate_ind_channel
#    else:
    ac_kernels = param_dict['Layer_%d/Slice_%d/ac_kernels' % (layer_idx, channel_idx)]

    # transform
    ## calc DC response: i.e. dc
    ## calc AC response
    num_channels = ac_kernels.shape[-1]
    dc = dc * num_channels * 1.0 / np.sqrt(num_channels)
    ac = np.matmul(sample_patches_centered, np.transpose(ac_kernels))
    transformed = np.concatenate([dc, ac], axis=1)
    del sample_patches_centered, ac_kernels
    gc.collect()
    #bin, print('Sample/cuboid_transformed_2d.shape:', transformed.shape)
    
    if train_flag:
        # Compute bias term for next layer
        bias = np.max(LA.norm(transformed, axis=1)) * np.ones(transformed.shape[-1])
    
    # Reshape back as a 4-D feature map
    feature_channel = np.array(transformed)
    sample_channel = feature_channel.reshape((output_n, output_h, output_w, -1)) # -> [num, h, w, c]
    sample_channel = np.moveaxis(sample_channel, 3, 1) # -> [num, c, h, w]
    
    if train_flag:
        return sample_channel, bias
    else:
        return sample_channel #, intermediate_ind_channel#, output_h, output_w


#%%
def multi_saab_chl_wise(sample_images_init, 
                        stride, 
                        kernel_size, 
                        dilate,
                        num_kernels,
                        energy_perc_thre, # int(num_nodes) or float(presE, var) for nodes growing down
                        padFlag,
                        recFlag = True,
                        collectFlag = False,
                        init_bias = 0):
    '''
    Do the Saab "training".
    the length should be equal to kernel_sizes.
    @Args:
        sample_images_init(np.array, np.float64),[num_images, channel, height, width]
        stride(list, int): stride for each stage
        kernel_size(list, int): subspace size for each stage, the length defines how many stages conducted
        dilate(list, int): dilate for each stage
        num_kernels(list, float or int): used in pca for number of valid AC components preservation. int w.r.t. number, float w.r.t. cumulative energy
        energy_perc_thre(float < 1): energy percantage threshold for a single node, if global ene perc < eneray_perc_threshold, then stop splitting
                                     only work for AC nodes
        padFlag(list, bool): indicator of padding for each stage
        recFlag(boolean): If true, the AC kernel number is limited by the AC feature matrix rank. This is necessary for reconstruction.
    @Returns: 
        pca_params(dict): residue saab transform parameters
        feature_all: a list of feature (all nodes) at each stage
        sample_images[num_images, channel, height, width]: feature at last stage
    '''
    num_layers = len(kernel_size)
    pca_params = {}
    
#    pca_params['num_layers'] = num_layers
    pca_params['stride'] = stride
    pca_params['kernel_size'] = kernel_size
    pca_params['dilate'] = dilate
    pca_params['num_kernels'] = num_kernels
    pca_params['energy_perc_thre'] = energy_perc_thre
    pca_params['padFlag'] = padFlag
    pca_params['recFlag'] = recFlag
    
    sample_images = sample_images_init.copy()
    pca_params['Layer_-1/h'] = sample_images.shape[-2]
    pca_params['Layer_-1/w'] = sample_images.shape[-1]
    pca_params['Layer_-1/intermediate_ind'] = np.ones(sample_images.shape[1])
    #pca_params['Layer_-1/Slice_0/bias'] = init_bias
    pca_params['Layer_-1/bias'] = np.array([init_bias]*sample_images.shape[1])
    #pca_params['Layer_-1/bias'] = np.array([init_bias])

    feature_all = []
    
    # for each layer
    i_valid = -1
    for i in range(num_layers):
        print('--------stage %d --------' % i)
        print('Sample/cuboid_previous.shape:', sample_images.shape)
        
        # prepare for next layer
        intermediate_ind_layer = pca_params['Layer_%d/intermediate_ind' % (i-1)]  # float list
        bias_layer = pca_params['Layer_%d/bias' % (i-1)]
        
        if np.sum(intermediate_ind_layer) > 0:
            i_valid += 1
            intermediate_index_layer = np.where(intermediate_ind_layer > pca_params['energy_perc_thre'])[0]
            sample_images = sample_images[:, intermediate_index_layer, :, :]
#            print('intermediate_ind_layer:', intermediate_index_layer)
            print('Sample/cuboid_forNextStage.shape:', sample_images.shape)
            
            # Maxpooling
            if i > 0:
                sample_images = block_reduce(sample_images, (1, 1, 2, 2), np.max)
                print('Sample/max_pooling.shape:', sample_images.shape)
                
            [n, num_node, h, w] = sample_images.shape
            
            feature_layer = [] # (unsorteed) features in this new hop from previous sorted feat.
            intermediate_ind_layer_new = []
            num_node_list = []
            bias_layer_new = []
            
#            print(bias_layer[intermediate_index_layer])

            
            # for each channel
            for channel_idx, intermediate_ene, bias in zip(np.arange(num_node), intermediate_ind_layer[intermediate_index_layer], bias_layer[intermediate_index_layer]):  # only partial samples going down
#                print('##### intermediate channel %d #####' % channel_idx)
                # Create patches in this channel
                sample_chl = sample_images[:, channel_idx:channel_idx+1, :, :]
#                sample_chl = np.expand_dims(sample_chl, axis=1)
                # extract filter and store in pca_params
                feature_channel, bias_chl = conv_with_bias(i, channel_idx,
                                                 sample_chl, bias,
                                                 pca_params, 
                                                 train_pack=intermediate_ene, train_flag=True)
                intermediate_ind_channel = pca_params['Layer_%d/Slice_%d/intermediate_ind' % (i, channel_idx)]
                # collect all slice feature in this layer
                feature_layer.append(feature_channel)
                intermediate_ind_layer_new.append(intermediate_ind_channel)
                num_node_list.append(len(intermediate_ind_channel)) # include DC, include small AC
                bias_layer_new.append(bias_chl)
                # end with all valid parent nodes (ALL CHANNELS)
                

            # one layer summary
            sample_images = np.concatenate(feature_layer, axis=1)  # only control features for further growing
            intermediate_ind_layer_new = np.concatenate(intermediate_ind_layer_new, axis=0)
            num_node_list = np.array(num_node_list)
            bias_layer_new = np.concatenate(bias_layer_new)
            pca_params['Layer_%d/intermediate_ind' % i] = intermediate_ind_layer_new # float list
            pca_params['Layer_%d/num_node_list' % i] = num_node_list
            pca_params['Layer_%d/bias' % i] = bias_layer_new
            pca_params['Layer_%d/h'% i] = sample_images.shape[-2]
            pca_params['Layer_%d/w'% i] = sample_images.shape[-1]
            print('Sample/cuboid_toNextStage.shape:', sample_images.shape)
#            print()
            
            if collectFlag:
                feature_all.append(sample_images)
            # end with ONE LAYER conditionally
        # end with ONE LAYER
        
#    #### adjust last layer intermediate_ind to be all none zero
#    intermediate_ind_layer_last = pca_params['Layer_%d/intermediate_ind' % i_valid]
#    intermediate_ind_layer_last[intermediate_ind_layer_last==0] = -1
#    pca_params['Layer_%d/intermediate_ind' % i_valid] = intermediate_ind_layer_last
    pca_params['num_layers'] = i_valid+1
    return pca_params, feature_all, sample_images



# feature generation
def inference_chl_wise(pca_params_init,
                        sample_images_init, intermediate_flag,
                        current_stage, target_stage,
                        collectFlag=True):
    '''
    Based on pca_params, generate saab feature for target_stage from current_stage
    stage index: -1(initial image), 0(1st saab), 1(2nd saab), ...
    @Args:
        pca_params(dict)
        sample_images_init(np.array, np.float64)[num,c,h,w]: 4-D feature of current_stage, of only intermediate nodes or all nodes (cannot be only leaf nodes)
        intermediate_flag(bool): [True] only gives intermediate nodes, or [False] gives all nodes
        current_stage(int): current stage index
        target_stage(int): target stage index
    @Returns:
        pca_params: modified c/w Saab filters on spatial sizes based on testing data, spectral operation is preserverd
        feature_all(list, np.array, np.float64)[sample_images0, sample_images1, ...]: 
            4-D feature of each stage in the processtarget_stage (exclude current_stage, include current_stage)
        sample_images(np.array, np.float64)[num, c, h, w]: 4-D feature of target_stage
    '''
    pca_params = pca_params_init.copy()
    
    # custermize spatial info
    sample_images = sample_images_init.copy()
    pca_params['Layer_%d/h'% current_stage] = sample_images.shape[-2]
    pca_params['Layer_%d/w'% current_stage] = sample_images.shape[-1]

    intermediate_ind_layer = pca_params['Layer_%d/intermediate_ind' % current_stage] # float list
    num_intermediate_init = len(np.where(intermediate_ind_layer > pca_params['energy_perc_thre'])[0])
    if intermediate_flag:
        assert num_intermediate_init == sample_images.shape[1]
    else:
        assert len(intermediate_ind_layer) == sample_images.shape[1]
    
    feature_all = []
    # for each layer
    for i in range(current_stage+1, target_stage+1, 1):
        print('--------stage %d --------' % i)
        print('Sample/cuboid_previous.shape:', sample_images.shape)
        
        intermediate_ind_layer = pca_params['Layer_%d/intermediate_ind' % (i-1)] # float list
        intermediate_index_layer = np.where(intermediate_ind_layer > pca_params['energy_perc_thre'])[0]
        bias_layer = pca_params['Layer_%d/bias' % (i-1)]
        # prepare for next layer, take out intermediate nodes
        if i == current_stage+1 and (intermediate_flag): # current_stage, only gives intermediate nodes
            pass
        else: # else gives all nodes
            sample_images = sample_images[:, intermediate_index_layer, :, :]
#            print('intermediate_ind_layer:', intermediate_index_layer)
#            print('Sample/cuboid_forNextStage.shape:', sample_images.shape)
        
        if sample_images.shape[1] > 0:
            # Maxpooling
            if i > 0:
                sample_images = block_reduce(sample_images, (1, 1, 2, 2), np.max)
                print('Sample/max_pooling.shape:', sample_images.shape)
                
            [n, num_node, h, w] = sample_images.shape
            feature_layer = []
    
            for channel_idx, bias in zip(np.arange(num_node),bias_layer[intermediate_index_layer]):
#                print('##### intermediate channel %d #####' % channel_idx)
                # Create patches in this slice
                sample_chl = sample_images[:, channel_idx, :, :]
                sample_chl = np.expand_dims(sample_chl, axis=1)
                
                feature_channel = conv_with_bias(i, channel_idx,
                                                 sample_chl, bias,
                                                 pca_params, 
#                                                 [kernel_size[i], stride[i], dilate[i], num_kernels[i], padF],
                                                 train_flag = False)
                # collect all slice feature in this layer
                feature_layer.append(feature_channel)
                
                # end with all valid parent nodes (ALL CHANNELS)
                
            # one layer summary
            sample_images = np.concatenate(feature_layer, axis=1)  # only control features for further growing
            pca_params['Layer_%d/h'% i] = sample_images.shape[-2]
            pca_params['Layer_%d/w'% i] = sample_images.shape[-1]
            print('Sample/cuboid_toNextStage.shape:', sample_images.shape)
#            print()
            
            if collectFlag:
                feature_all.append(sample_images)
            # end with ONE LAYER conditionally
        # end with ONE LAYER
        
    ### last layer  intermediate_ind is not used
    return pca_params, feature_all, sample_images #, sample_images # last stage feature [num, c, h, w]







#%%
if __name__ == "__main__":

    init_images = np.random.rand(5, 1, 33, 44) # (num_img, num_channel, img_h, img_w)
    
    print('\n######  Saak Feature Extraction  ######\n')
    params, feature_all, feature_last = multi_saab_chl_wise(init_images, 
                                                            [3], # stride, 
                                                            [3], # kernel_size, 
                                                            [1], #dilate,
                                                            ['full'], #num_kernels,
                                                            0.125, #energy_perc_thre, # int(num_nodes) or float(presE, var) for nodes growing down
                                                            padFlag = [False],
                                                            recFlag = True,
                                                            collectFlag = True)


    print('\n###### check inference_chl_wise ######\n')
    __, __, feature1 = inference_chl_wise(params,
                                          init_images, True,
                                          -1, 0,
                                          collectFlag=False)
    print(set((feature_last.reshape(-1) == feature1.reshape(-1)).tolist()))

    print()
