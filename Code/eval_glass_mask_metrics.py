# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 21:51:52 2023
command: !python eval_glass_mask_metrics.py -p "path for prediction dir" -gt "path for gt dir" -th 64
"""

import os, sys
import cv2
import glob
import argparse
import numpy as np
from sklearn.metrics import fbeta_score, mean_absolute_error, balanced_accuracy_score, recall_score, auc, roc_curve

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--prediction", type=str, required=True)
parser.add_argument("-gt", "--groundtruth", type=str, required=True)
parser.add_argument("-th", "--thresh", type=int, default=0)
args = parser.parse_args()

def reshape_mask(mask, thresh):
    """    

    Parameters
    ----------
    mask : 
        masked obj val > 0.
    thresh : uint8
        DESCRIPTION.

    Returns
    -------
    re_mask : uint8 image

    """
    re_mask = np.where((mask > thresh), 255, np.zeros(mask.shape, dtype=np.uint8))
    re_mask = re_mask.reshape(-1)
    return re_mask

def calculate_iou(gt_mask, pred_mask):
    # If either set of masks is empty return empty result
    if gt_mask.shape[-1] == 0 or pred_mask.shape[-1] == 0:
        return np.zeros((gt_mask.shape[-1], pred_mask.shape[-1]))
    
    # intersection and union
    union = cv2.bitwise_or(gt_mask, pred_mask)
    intersection = cv2.bitwise_and(gt_mask, pred_mask)
    area_union = np.sum(union)
    area_intersection = np.sum(intersection)
    iou = area_intersection / area_union
    return iou

def calculat_recall(gt_mask, pred_mask):
    # If either set of masks is empty return empty result
    if gt_mask.shape[-1] == 0 or pred_mask.shape[-1] == 0:
        return np.zeros((gt_mask.shape[-1], pred_mask.shape[-1]))
       
    intersection = cv2.bitwise_and(gt_mask, pred_mask)
    area_intersection = np.sum(intersection==255)
    area_gt = np.sum(gt_mask==255)
    recall = area_intersection / area_gt
    return recall

if __name__ == "__main__":
    gt_file_list = []
    pred_file_list = []
    EXT_GT_IMG = ['.png']
    EXT_DEPTH_IMG = ['.png']
    for ext in EXT_GT_IMG:        
        gt_file_list += (sorted(glob.glob(os.path.join(args.groundtruth, '*' + ext))))
    for ext in EXT_DEPTH_IMG:
        pred_file_list += (sorted(glob.glob(os.path.join(args.prediction, '*' + ext))))
    assert len(gt_file_list) == len(pred_file_list), (
            'number of rgb ({}) and depth images ({}) are not equal'.format(len(gt_file_list), len(pred_file_list)))
    
    IOU = [] 
    FB = []
    MAE = []
    BER = []
    RECALL = []
    AUC = []
    for i in range(len(gt_file_list)):
        gt_np = cv2.imread(gt_file_list[i], cv2.IMREAD_GRAYSCALE)
        gt_np = cv2.resize(gt_np, (320, 256))
        pd_np = cv2.imread(pred_file_list[i], cv2.IMREAD_GRAYSCALE)
        pd_np = cv2.resize(pd_np, (320, 256))
        
        gt = reshape_mask(gt_np, args.thresh)
        pd = reshape_mask(pd_np, args.thresh)
        
        recall = calculat_recall(gt, pd)
        reshape_pd = 0
        # if (recall < 0.5):
        #     pd = reshape_mask(pd_np, 0)
        #     recall = calculat_recall(gt, pd)
        RECALL.append(recall) 
        
        iou = calculate_iou(gt, pd)
        # if (iou < 0.5):
        #     pd = reshape_mask(pd_np, 0)
        #     iou = calculate_iou(gt, pd)        
        IOU.append(iou)        
        
        fb = fbeta_score(gt, pd, average='binary', pos_label=255, beta=0.3)
        FB.append(fb)        
        mae = mean_absolute_error(gt, pd)
        MAE.append(mae)
        bac = balanced_accuracy_score(gt, pd)
        ber = (1-bac)*100
        BER.append(ber)
        fpr, tpr, thresholds = roc_curve(gt, pd, pos_label=255)
        auc_score = auc(fpr, tpr)
        AUC.append(auc_score)

    IOU = np.mean(IOU)
    FB = np.mean(FB)
    MAE = np.mean(MAE)
    BER = np.mean(BER)
    RECALL = np.mean(RECALL)
    AUC = np.mean(AUC)
    
    print (f'IOU : {IOU}')    
    print (f'FB : {FB}')
    print (f'MAE : {MAE}')
    print (f'BER : {BER}')
    print (f'AUC : {AUC}')
    print (f'RECALL : {RECALL}')