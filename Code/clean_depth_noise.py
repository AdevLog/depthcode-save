# -*- coding: utf-8 -*-
"""
Created on Mon May  1 22:48:55 2023

txt file for Matterport3D structure
    scene name: 2t7WUuJeko7
        image type: undistorted_color_images
            image: resize_06addff1d8274747b7a1957b2f03b736_i1_4.jpg
            
You can modify the file path on your own.
test_list_filename, clean_depth_directory
"""
import sys, os
import cv2
import numpy as np

def clean_mask(image, iterations=2):
    """
    input: 1 channel grayscale. dtype=np.uint8, Shape=(H, W, 3) or Shape=(H, W)
    output: 1 channel grayscale. dtype=np.uint8, Shape=(H, W)
            white is background, black is mask
    """
    if len(image.shape) == 3:        
        gray = image[:,:,0]
    elif len(image.shape) == 2:
        gray = image
    else:
        raise ValueError('The shape of the tensor should be (H, W, 3) or (H, W). ' +
                         'Given shape is {}'.format(image.shape))
    pad = iterations+1
    pad_image = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_CONSTANT, None, value = 0)     
    mask = np.zeros(pad_image.shape, dtype=np.uint8)
    mask[pad_image>0] = 255 # white is foreground objects
    for i in range(iterations):
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)    
        drawC = cv2.drawContours(mask, contours, -1, (0,0,0), 1) # enlarge black holes
    crop_image = drawC[pad:pad+image.shape[0], pad:pad+image.shape[1]]
    print("crop_image",drawC.shape)
    return crop_image

if __name__ == "__main__":    
    # txt in the same path with depth image
    test_list_filename = r"G:/matterport3D/M3D_mask/raw_depth_140.txt"
    # os.path.join(Path1,Path2,Path3)    
    dirname = os.path.dirname(test_list_filename)
    print("dirname:", dirname)
    
    with open(test_list_filename, 'r') as f:
        content = f.read().splitlines()   
    
    #==========read depth file==========#
    for ele in content:
        depth_img_path = os.path.join(dirname,ele)     
        depth_img = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)
        print("depth_img shape:", depth_img.shape)     
        #==========clean depth ==========#
        mask = clean_mask(depth_img)
        depth_img[mask==0] = 0
        #==========make clean depth dir==========#
        # left, _, right = ele.split('/') # left: scene_name, _: undistorted_color_images , right: resize_06addff1d8274747b7a1957b2f03b736_i1_4.jpg        
        # clean_depth_filename = os.path.join(r"G:\Python Project\Glass Surface Detection", 'clean_depth', left, 'undistorted_depth_images_clean', right.split('.')[0]+'_clean.png')
        left, right = ele.split('/') # left: directory_name, right: resize_06addff1d8274747b7a1957b2f03b736_i1_4.jpg        
        clean_depth_directory = r"G:/matterport3D/M3D_mask/"
        clean_depth_filename = os.path.join(clean_depth_directory, right.split('.')[0]+'_clean.png')
        
        try:
            os.makedirs(clean_depth_directory, exist_ok = True)
            print("Directory '%s' created successfully" %clean_depth_directory)
        except OSError as error:
            print("Directory '%s' can not be created")
        
        # print("left:", left)
        # print("light_boundary_f:", light_boundary_filename)
        depth_img = depth_img.astype(np.uint16)
        cv2.imwrite(clean_depth_filename, depth_img)
#         cv2.imshow('depth_img', depth_img)
#         cv2.imshow('crop_img', crop_img)
#         cv2.waitKey(0)
        
# cv2.destroyAllWindows()