# -*- coding: utf-8 -*-
"""
Created on Wed May 24 04:52:01 2023
modified from Indoor Depth Completion with Boundary Consistency and Self-Attention
https://github.com/tsunghan-wu/Depth-Completion/blob/master/depth_completion/eval.py
"""

import os, sys
import cv2
import glob
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim

def miss(output_depth, render_depth, mask, t):
    output_over_render = np.divide(output_depth, render_depth, out=np.zeros_like(output_depth), where=render_depth!=0)
    render_over_output = np.divide(render_depth, output_depth, out=np.zeros_like(render_depth), where=output_depth!=0)
    output_over_render[np.isnan(output_over_render)] = 0
    render_over_output[np.isnan(render_over_output)] = 0
    miss_map = np.maximum(output_over_render, render_over_output)
    hit_rate = np.sum(miss_map[~mask] < t).astype(float) 
    return hit_rate

def depth_rel(output_depth, render_depth, mask):
    err = np.abs(render_depth-output_depth)
    diff = np.divide(err, render_depth, out=np.zeros_like(err), where=render_depth!=0)
    diff[mask] = 0 # remove nan
    return diff[~mask].reshape(-1)

if __name__ == "__main__":

    raw_input_dir = r"G:/matterport3D/M3D_mask" # To exclude regions not in holes.
    gt_dir = r"G:/matterport3D/M3D_mask"
    depth_dir = r"G:/matterport3D/M3D_mask"
    input_file_list = []
    gt_file_list = []
    depth_file_list = []
    EXT_RAW_IMG = ['.png']
    EXT_GT_IMG = ['_mesh_depth.png']
    EXT_DEPTH_IMG = ['.png']
    for ext in EXT_RAW_IMG:        
        input_file_list += (sorted(glob.glob(os.path.join(raw_input_dir, '*' + ext))))
    for ext in EXT_GT_IMG:        
        gt_file_list += (sorted(glob.glob(os.path.join(gt_dir, '*' + ext))))
    for ext in EXT_DEPTH_IMG:
        depth_file_list += (sorted(glob.glob(os.path.join(depth_dir, '*' + ext))))
    assert len(gt_file_list) == len(depth_file_list), (
            'number of gt ({}) and depth images ({}) are not equal'.format(len(gt_file_list), len(depth_file_list)))

    REL = []
    L1 = []
    RMSE = []
    SSIM = []
    delta_105 = []
    delta_110 = []
    delta_125_1 = []
    delta_125_2 = []
    delta_125_3 = []
    N = 0

    for i in range(len(gt_file_list)):    

        
        input_depth = cv2.imread(input_file_list[i], cv2.IMREAD_UNCHANGED)
        gt_np = cv2.imread(gt_file_list[i], cv2.IMREAD_UNCHANGED)
        pd_np = cv2.imread(depth_file_list[i], cv2.IMREAD_UNCHANGED)
        
        # scale the depth image from uint16 to denote the depth in meters (float32)
        gt_np= gt_np.astype(np.float32)
        pd_np= pd_np.astype(np.float32)
        gt_np /= 4000.00
        pd_np /= 4000.00
        
        #221213 add
        gt_np[np.isnan(gt_np)] = 0.0
        gt_np[np.isinf(gt_np)] = 0.0
        
        #230702 add
        input_depth[np.isnan(input_depth)] = 1.0
        input_depth[np.isinf(input_depth)] = 1.0
        
        invalid_mask = np.where((gt_np == 0.0), 255, 0)
        invalid_mask = np.where((input_depth > 0.0), 255, invalid_mask)
        
        
        mask = (invalid_mask > 0.0) # To calculate only within the hole filling region
        # mask = (gt_np == 0.0 ) # To calculate whole region
        pd_np[mask] = 0.0
        gt_np[mask] = 0.0
        

        # calculate valid pixels
        n = 256 * 320 - np.sum(mask) 
        N += n
        
        # calculate hit
        delta_105.append(miss(pd_np, gt_np, mask, 1.05))
        delta_110.append(miss(pd_np, gt_np, mask, 1.10))
        delta_125_1.append(miss(pd_np, gt_np, mask, 1.25))
        delta_125_2.append(miss(pd_np, gt_np, mask, 1.25**2))
        delta_125_3.append(miss(pd_np, gt_np, mask, 1.25**3))

        # calculate mse
        RMSE.append(((gt_np-pd_np)**2))
        
        # calculate L1
        L1.append(np.abs(gt_np-pd_np))

        # calculate rel
        rel_err = depth_rel(pd_np, gt_np, mask)
        REL += (list(rel_err))

        # calculate ssim        
        ski_ssim = ssim(pd_np, gt_np, data_range=17)
        SSIM.append(ski_ssim)
    
    SSIM = np.mean(SSIM)
    L1 = np.mean(L1)
    RMSE = np.sqrt(np.mean(RMSE))
    # REL = np.median(np.array(REL).reshape(-1))
    REL = np.mean(np.array(REL).reshape(-1))
    delta_105 = np.sum(delta_105) / N
    delta_110 = np.sum(delta_110) / N
    delta_125_1 = np.sum(delta_125_1) / N
    delta_125_2 = np.sum(delta_125_2) / N
    delta_125_3 = np.sum(delta_125_3) / N
    
    print (f'L1(MAE) : {L1:.3f}')    
    print (f'RMSE : {RMSE:.3f}')
    print (f'REL : {REL:.3f}')
    print (f'SSIM : {SSIM:.3f}')
    print (f'1.05 : {delta_105:.3f}')
    print (f'1.10 : {delta_110:.3f}')
    print (f'1.25 : {delta_125_1:.3f}')
    print (f'1.25^2 : {delta_125_2:.3f}')
    print (f'1.25^3 : {delta_125_3:.3f}')