#!/usr/bin/env python3

import os
import glob
import numpy as np

from api import utils as api_utils


depth_dir = os.path.join(r'G:/matterport3D/M3D_mask/')
# ext = '*.npy'
new_ext = '.exr'

EXT_IMG = ['output.npy', '_output_clean.npy']
depth_files_npy = []
for ext in EXT_IMG:
    depth_files_npy += (sorted(glob.glob(os.path.join(depth_dir, '*' + ext))))
    
for depth_file in depth_files_npy:    
    depth_img = np.load(depth_file)   
    # depth_img = np.load(depth_files_npy[depth_file])
    filename = os.path.basename(depth_file)
    filename_no_ext = os.path.splitext(depth_file)[0]
    print(filename, depth_img.shape)

    new_filename = os.path.join(filename_no_ext + new_ext)
    depth_img = np.stack((depth_img, depth_img, depth_img), axis=0)
    api_utils.exr_saver(new_filename, depth_img, ndim=3)