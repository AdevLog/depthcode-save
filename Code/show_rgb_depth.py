"""
讀取資料夾中多張深度圖並顯示內容及顏色(雜訊)
"""
import os
import glob
import numpy as np
import cv2

from api import utils


depth_dir = os.path.join(r'G:/matterport3D/M3D_mask/')
ext = '*'+'.png'
new_ext = '_new.exr'


depth_files_png = glob.glob(os.path.join(depth_dir, ext))
print(depth_files_png)

for depth_file in depth_files_png:
    depth_img = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
    depth_img = np.float32(depth_img)
    # depth_img = depth_img / 1000 #scannet
    depth_img = depth_img / 4000 #Matterport3D
    print("max depth", np.max(depth_img)) #圖片最深的深度值
    print("depth_img shape",depth_img.shape)   
    if len(depth_img.shape) == 2:
        depth_rgb = utils.depth2rgb(depth_img, min_depth=0.0,
                                          max_depth=float(np.max(depth_img))+1, color_mode=cv2.COLORMAP_JET,
                                          reverse_scale=True)

        # cv2.imshow('depth', depth_rgb)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
            
        cv2.imwrite(os.path.splitext(depth_file)[0] + '_rgb.jpg', depth_rgb)