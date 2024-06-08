"""
!python convert_exr_depth_to_png.py -p "Path to dir" -d 'Max depth'
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import imageio

sys.path.append('..')
from api import utils as api_utils


parser = argparse.ArgumentParser(description='Convert folder of depth EXR images to png')
parser.add_argument('-p', '--path', required=True, help='Path to dir', metavar='path/to/dir')
parser.add_argument('-d', '--depth', required=True, type=float, help='Max depth', metavar='0.0')
args = parser.parse_args()

dir_exr = args.path
max_depth = float(args.depth)
depth_file_list = sorted(glob.glob(os.path.join(dir_exr, '*.exr')))

# COLORMAP = cv2.COLORMAP_TWILIGHT_SHIFTED
COLORMAP = cv2.COLORMAP_JET

print('Converting EXR files to RGB files in dir: {}'.format(dir_exr))
for depth_file in depth_file_list:
    depth_img = api_utils.exr_loader(depth_file, ndim=1)
    depth_img_rgb = api_utils.depth2rgb(depth_img, min_depth=0.0, max_depth=float(np.max(depth_img))+1, color_mode=COLORMAP,
                                        reverse_scale=False, dynamic_scaling=True)
    
    

    depth_filename_rgb = os.path.splitext(depth_file)[0] + '_rgb.png'
    imageio.imwrite(depth_filename_rgb, depth_img_rgb)
    

    depth_filename = os.path.splitext(depth_file)[0] + '.png'
    depth_u8_filename = os.path.splitext(depth_file)[0] + '_uint8.png'
    cv2.imwrite(depth_filename, depth_img.astype(np.uint16))
    cv2.mwrite(depth_u8_filename, depth_img)

    print('Converted image {}'.format(os.path.basename(depth_file)))