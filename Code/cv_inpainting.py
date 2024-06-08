# -*- coding: utf-8 -*-
"""
Created on Sun May 14 17:26:37 2023

"""
import cv2
import numpy as np

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
        print("depth_image shape",depth_image.shape)    
        mask = np.where(depth_image, 0, 255).astype(np.uint8) # Fill areas where value = 255
        dst = depth_image
        dst = cv2.inpaint(dst,mask,25,cv2.INPAINT_TELEA)
        # cv2.imshow('depth_image', depth_image)
        # cv2.imshow('mask', mask)
        # cv2.imshow('dst', dst)
            
        cv2.imwrite(os.path.join(destination_dir, filename.split('.')[0] + '_inpainted.png'), dst.astype(np.uint16))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()