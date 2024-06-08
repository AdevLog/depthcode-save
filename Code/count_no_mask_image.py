# -*- coding: utf-8 -*-
"""
glass area threshold: 0, 64, 128, 192, 255
"""

import numpy as np
import os
import glob
import cv2

if __name__ == "__main__":
    img_dir = r"G:/matterport3D/M3D_mask"
    file_list = []
    EXT_IMG = ['.png']
    for ext in EXT_IMG:
        file_list += (sorted(glob.glob(os.path.join(img_dir, '*' + ext))))
        
    cnt = 0
    thr = 0 # 0, 64, 128, 192, 255
    for i in range(len(file_list)):
        image = cv2.imread(file_list[i], cv2.IMREAD_UNCHANGED)
        # if (np.max(image) == 255):
        if (np.max(image) > thr):
        # if (np.sum(image > 0) > 82): # GSDS
            cnt = cnt + 1
            
    print("glass count: ", cnt)
        