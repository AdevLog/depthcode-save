# -*- coding: utf-8 -*-
"""
Created on Wed May 17 08:20:41 2023
"""

import cv2
import numpy as np
from scipy.interpolate import griddata

def fill_hole(image, mask):
    # Create a copy of the image
    filled_image = np.copy(image)
    
    # Get the dimensions of the image
    height, width = image.shape
    
    # Create a grid of coordinates
    x, y = np.meshgrid(range(width), range(height))
    
    # Flatten the coordinates and mask
    x_flat = x.flatten()
    y_flat = y.flatten()
    mask_flat = mask.flatten()
    
    # Filter the coordinates and mask to get only the hole points
    hole_points = np.where(mask_flat == 255)
    x_hole = x_flat[hole_points]
    y_hole = y_flat[hole_points]
    
    # Filter the coordinates and mask to get only the valid points
    valid_points = np.where(mask_flat == 0)
    x_valid = x_flat[valid_points]
    y_valid = y_flat[valid_points]
    
    # Get the pixel values for the valid points
    valid_values = filled_image.flatten()[valid_points]
    
    # Interpolate the valid points to get the filled values for the hole points
    # filled_values = griddata((x_valid, y_valid), valid_values, (x_hole, y_hole), method='nearest')
    filled_values = griddata((x_valid, y_valid), valid_values, (x_hole, y_hole), method='linear')
    
    # Set the filled values in the hole points
    filled_image[y_hole, x_hole] = filled_values
    
    return filled_image

if __name__ == "__main__":    
    depth_dir = r"G:/matterport3D/M3D_mask"
    destination_dir = r"G:/matterport3D/M3D_mask"
    
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    depth_file_list = []
    EXT_DEPTH_IMG = ['.png']
    for ext in EXT_DEPTH_IMG:
        depth_file_list += (sorted(glob.glob(os.path.join(depth_dir, '*' + ext))))
        
    for i in range(len(depth_file_list)):
        filename = os.path.basename(depth_file_list[i])
        depth_image = cv2.imread(depth_file_list[i], cv2.IMREAD_UNCHANGED)
        print("depth_image.shape",depth_image.shape)
        valid_depth = (depth_image > 0)
        mask = np.where(valid_depth, 0, 255) # Fill areas where missing values
                
        
        fill_image = fill_hole(depth_image, mask)                
        cv2.imwrite(os.path.join(destination_dir, filename.split('.')[0] + '_griddata_clean.png'), fill_image.astype(np.uint16))
        # cv2.imshow('filled_image',filled_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()