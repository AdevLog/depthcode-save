"""
OpenCV GrabCut: https://pyimagesearch.com/2020/07/27/opencv-grabcut-foreground-segmentation-and-extraction/

取亮度差異小於10的區域(亮度變化小)，去除該區域後，取剩下區域(亮度變化大)的亮度邊緣
"""
import numpy as np
import argparse
import time
import cv2
import os
import glob
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
	default=os.path.sep.join(["images", "adrian.jpg"]),
	help="path to input image that we'll apply GrabCut to")
ap.add_argument("-c", "--iter", type=int, default=10,
	help="# of GrabCut iterations (larger value => slower runtime)")
args = vars(ap.parse_args())

def calculat_recall(masks1, masks2):
    """    

    Parameters
    ----------
    masks1 : gt
        DESCRIPTION.
    masks2 : pred
        DESCRIPTION.

    Returns
    -------
    TYPE
        precision.

    """
    # If either set of masks is empty return empty result
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))
    # # flatten masks and compute their areas
    # masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    # masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    # area1 = np.sum(masks1, axis=0)
    # area2 = np.sum(masks2, axis=0)
    
    # intersections and union
    intersections = cv2.bitwise_and(masks1, masks2)
    area_intersection = np.sum(intersections==255)
    area_mask1 = np.sum(masks1==255)
    precision = area_intersection / area_mask1
    # union = area1[:, None] + area2[None, :] - intersections
    # overlaps = intersections / union

    return precision

def erode_mask(mask):
    inverse_mask = cv2.bitwise_not(mask)
    contours, hierarchy = cv2.findContours(inverse_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    draw_contour = cv2.drawContours(inverse_mask, contours, -1, (255,255,255), -1).astype("uint8")
    draw_contour = cv2.drawContours(draw_contour, contours, -1, (255,255,255), 2).astype("uint8")
    inverse_mask = cv2.bitwise_not(draw_contour)
    return inverse_mask

def enlarge_mask(image, thickness):
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
    mask = np.zeros(gray.shape, dtype=np.uint8)
    # mask[gray>0] = 255 # white is foreground objects
    iterations = 1
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for i in range(iterations):   
        draw_contour = cv2.drawContours(mask, contours, -1, (255,255,255), -1)
        draw_contour = cv2.drawContours(draw_contour, contours, -1, (255,255,255), thickness) # enlarge mask
    return draw_contour

def check_high_contrast(image, mask):
    """
    If there is a lack of seed points, it is necessary to check whether the image exhibits high contrast.  

    Parameters
    ----------
    image :
        It can be any input, which will be normalize to a uint8 file.
    mask : uint8
        mask of missing depth.

    Returns
    -------
    high_contrast : boolean
        DESCRIPTION.

    """
    img_gray = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)    
    # Canny Edge Detection
    edges = cv2.Canny(img_gray, 100, 250) # Canny Edge Detection    
    erosion_mask = erode_mask(mask) # avoid contour line    
    check_mask = np.where(erosion_mask, edges, 0) # edges is 255
    # cv2.imshow('edges', edges)
    # cv2.imshow('erosion_mask', erosion_mask)
    # cv2.imshow('check_mask', check_mask)
    check_val = np.max(check_mask)
    high_contrast = True
    if check_val > 0:
        high_contrast = True
        print("The mask region has high contrast")
    else:
        high_contrast = False
        # print("The mask region has adjacent pixels within the threshold distance.")
    return high_contrast

def grabCut_mask(img, mask):
    fgModel = np.zeros((1, 65), dtype="float")
    bgModel = np.zeros((1, 65), dtype="float")
    # apply GrabCut using the the bounding box segmentation method
    start = time.time()
    (mask_cut, bgModel, fgModel) = cv2.grabCut(image, mask, None, bgModel,
                                           fgModel, iterCount=args["iter"], mode=cv2.GC_INIT_WITH_MASK) #GC_INIT_WITH_RECT
    
    end = time.time()
    print("[INFO] applying GrabCut took {:.2f} seconds".format(end - start))
        
    # we'll set all definite background and probable background pixels
    # to 0 while definite foreground and probable foreground pixels are
    # set to 1
    output_mask = np.where((mask_cut == cv2.GC_BGD) | (mask_cut == cv2.GC_PR_BGD), 0, 1)
    # scale the mask from the range [0, 1] to [0, 255]
    output_mask = (output_mask * 255).astype("uint8")
    return output_mask

def trim_valid_mask_with_depth(image, depth, mask):
    # allocate memory for two arrays that the GrabCut algorithm internally
    # uses when segmenting the foreground from the background
    fgModel = np.zeros((1, 65), dtype="float")
    bgModel = np.zeros((1, 65), dtype="float")
                
    valid_mask = np.zeros(depth.shape, dtype=np.uint8)
    valid_glass_area = 0
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):        
        if cv2.contourArea(contours[i]) > 80: # contour area greater than 0.1% of image(about 81 pixels)
            draw_contour_area = cv2.drawContours(np.zeros(depth.shape, dtype=np.uint8), contours, i, (255,255,255), -1).astype("uint8")          
            # cv2.imshow("draw_contour_area", draw_contour_area)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()  
            
            valid_depth = np.where((draw_contour_area > 0), depth, 1).astype("uint8")
            # cv2.imshow("valid_depth", valid_depth.astype("uint16")) 
            
            # masked hole area = 255 (depth=0)
            valid_depth_area = np.where((valid_depth > 0), np.zeros(depth.shape, dtype=np.uint8), 255).astype("uint8")

            # adjust glass area
            valid_prob = calculat_recall(draw_contour_area, valid_depth_area)

            if (valid_prob <= 0.9) and (valid_prob > 0.1):
                invalid_depth = np.where((draw_contour_area > 0), depth, 0).astype("uint8")
                invalid_depth_mask = np.where((invalid_depth > 0), 1, 2).astype("uint8")
                # cut the invalid depth area (possible glass area)               
                invalid_depth_cut = grabCut_mask(image, invalid_depth_mask)
                valid_glass_area = cv2.bitwise_and(draw_contour_area, invalid_depth_cut)  
                    
            elif (valid_prob <= 0.1):
                # 去除深度零 計算剩餘mask區是否為完整平面深度
                # 此區域可能是牆角 或是包含很多雜物
                # 取邊緣 若有邊緣 代表深度不正確
                # mask那些深度不連續的
                contour_depth = np.where(draw_contour_area, depth, 0)
                # cv2.imshow("contour_depth", contour_depth)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows() 
                check_high_contrast_area = np.where((contour_depth > 0), 255, 0).astype("uint8")
                is_contrast = check_high_contrast(contour_depth, check_high_contrast_area)
                if (is_contrast == True):
                    print('=====is_contrast=====')                                   
                    valid_glass_area = cv2.bitwise_or(draw_contour_area, check_high_contrast_area)
                    # cv2.imshow("valid_glass_area", valid_glass_area.astype("uint8"))
                    # # valid_mask[check_high_contrast_area] == 0
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                else :
                    print('=====no contrast=====')
                    valid_glass_area = cv2.bitwise_and(valid_mask, valid_mask)
            #         valid_glass_area[1:2,1:2] = 255 # make small dot
            #         cv2.imshow("check_high_contrast_area", check_high_contrast_area.astype("uint8"))
            #         cv2.waitKey(0)
            #         cv2.destroyAllWindows()
            ##########################
            # 特別針對補滿的察看是否深度不連續
            ##########################
            else:
                # cut the valid depth area (possible glass area) 
                valid_depth_mask = np.where((valid_depth_area > 0), 1, 2).astype("uint8")
                valid_depth_cut = grabCut_mask(image, valid_depth_mask)
                # union depth hole and color mask
                valid_glass_area = cv2.bitwise_and(draw_contour_area, valid_depth_cut)
                
            valid_mask = cv2.bitwise_or(valid_mask, valid_glass_area).astype("uint8")
            # cv2.imshow("valid_mask", valid_mask.astype("uint8"))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
    return valid_mask

if __name__ == "__main__":
    # Specify the destination directory where the files will be copied
    img_dir = r"G:/matterport3D/M3D_mask/mask_20/color_20" # raw input
    depth_dir = r"G:/matterport3D/M3D_mask/mask_20/raw_20" # raw input
    destination_dir = r"G:/matterport3D/M3D_mask/mask_20/grabCut_20_01_90"
    
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    
    rgb_file_list = []
    depth_file_list = []
    EXT_RGB_IMG = ['.jpg']
    EXT_DEPTH_IMG = ['.png']
    for ext in EXT_RGB_IMG:
        rgb_file_list += (sorted(glob.glob(os.path.join(img_dir, '*' + ext))))
    for ext in EXT_DEPTH_IMG:
        depth_file_list += (sorted(glob.glob(os.path.join(depth_dir, '*' + ext))))
    
    for i in range(len(rgb_file_list)):
        filename = os.path.basename(rgb_file_list[i])
        print('filename', filename)
        # load the input image from disk and then allocate memory for the
        # output mask generated by GrabCut -- this mask should hae the same
        # spatial dimensions as the input image
        # image = cv2.imread(args["image"])        
        image = cv2.imread(rgb_file_list[i], cv2.IMREAD_UNCHANGED)
        # mask = np.zeros(image.shape[:2], dtype="uint8")
        depth = cv2.imread(depth_file_list[i], cv2.IMREAD_UNCHANGED)
        no_glass_area = 0
        zero_mask = np.zeros(depth.shape, dtype=np.uint8)
        glass_mask = np.zeros(depth.shape, dtype=np.uint8)
        
        # hsv轉換
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_copy = hsv.copy()
        h, s, v = cv2.split(hsv)
        bright_area = np.where((v > 250), 255, zero_mask).astype("uint8")
        hole_area = np.where((depth == 0), 255, zero_mask).astype("uint8")
        bright_hole = cv2.bitwise_and(bright_area, hole_area)
        hsv[bright_hole>0] = (50,50,50)
        # hsv[v>250] = (0,0,0) # 包含亮處
        hsv[h<50] = (255,255,255)
        hsv[h>100] = (255,255,255)
        lower_hsv = np.array([50, 50, 50])
        upper_hsv = np.array([100, 255, 255])
        hsv_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        # cv2.imshow("hsv_mask == 255", hsv_mask)
        # res = cv2.bitwise_and(image, image, mask=hsv_mask) # mask出來的彩色區域，黑色部分是0，只要and之後都會是黑
        # hsv_mask = np.where(hsv_mask == 255, 3, 2).astype("uint8") # (2) probable background, and (3) probable foreground
        
        dilated_v = cv2.dilate(hsv[:,:,2], np.ones((3,3), np.uint8))
        # cv2.imshow("v", v)
        # cv2.imshow("dilated_v", dilated_v)
        diff_light = cv2.absdiff(v, dilated_v) # 取邊緣 最大174
        # smooth_v = cv2.GaussianBlur(dilated_v, (401,401), 0)
        # # 0706
        # smooth_v = cv2.GaussianBlur(dilated_v, (5,5), 0)
        # division_v = cv2.divide(smooth_v, v, scale=255)
        # division_mask = np.zeros(image.shape[:2], dtype="uint8")
        # division_mask[division_v<200] = 255 # 255==white
        # # division_mask = cv2.bitwise_not(division_mask)
        # # division_mask = np.where(division_mask == 255, 3, 2).astype("uint8")        
        
        light_mask = np.where((diff_light < 10), 255, 0).astype("uint8")
        diff_light[diff_light<10] = 0    
        contrast = 5
        brightness = 0
        thick_light = cv2.addWeighted(diff_light, contrast, np.zeros(diff_light.shape, diff_light.dtype), 0, brightness)        
        # hsv_mask = np.where((thick_light > 0), 0, hsv_mask).astype("uint8")
        light_mask = np.where((thick_light > 0), 0, light_mask).astype("uint8")    
        # valid = (smooth_v > 180) & (smooth_v < 200) 
        # light_mask = np.where(valid , 255, 0).astype("uint8")
        
        valid_hsv_mask = trim_valid_mask_with_depth(image, depth, hsv_mask)
        valid_light_mask = trim_valid_mask_with_depth(image, depth, light_mask)
        
        if (np.max(valid_hsv_mask) > 0) and (np.max(valid_light_mask) > 0) :
            valid_color_mask = (valid_hsv_mask == 255) & (valid_light_mask == 255)
            color_and = np.where(valid_color_mask, 3, 2).astype("uint8")
        else:
            color_and = zero_mask
        if (np.max(valid_hsv_mask) > 0) or (np.max(valid_light_mask) > 0) :
            # color_mask = np.where((hsv_mask == 255) | (division_mask == 255) | (light_mask == 255), 3, 2).astype("uint8")
            color_or = np.where((valid_hsv_mask == 255) | (valid_light_mask == 255), 3, 2).astype("uint8")        
        else:
            color_or = zero_mask
        # color_and_view = np.where(valid_color_mask, 255, 0).astype("uint8")
        # print('color_mask shape', color_mask.shape)
        # color_mask_view = np.where((valid_hsv_mask == 255) | (valid_light_mask == 255), 255, 0).astype("uint8")        
        
        if (np.max(depth) > 40): # depth value greater than 1 cm
            depth_mask = np.where((depth == 0), 255, 0).astype("uint8") 
            valid_depth_mask = trim_valid_mask_with_depth(image, depth, depth_mask)
            trim_depth = np.where(valid_depth_mask == 255, 3, 2).astype("uint8")
        else:
            trim_depth = zero_mask   
        
        # if (np.max(color_and) == 3) and (np.sum(color_and==3) > 80):      
        if (np.sum(color_and==3) > 80): # contour area greater than 0.1% of image(about 81 pixels)
            output_color_and = grabCut_mask(image, color_and)            
       
        else: 
            output_color_and = zero_mask
        if (np.sum(color_or==3) > 80):
            # print('np sum',np.sum(color_or==3))
            output_color_or = grabCut_mask(image, color_or)

        else: 
            output_color_or = zero_mask
        if (np.sum(trim_depth==3) > 80):   
            print('np sum trim_depth',np.sum(trim_depth==3))            
            output_depth_mask = grabCut_mask(image, trim_depth)   
            # cv2.imshow("output_depth_mask", output_depth_mask)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        else: 
            output_depth_mask = zero_mask
        
        # cv2.imshow("depth_mask_100_cut", output_depth_mask_100_cut)
        valid_glass_area = 0
        valid_glass_area = cv2.bitwise_or(output_color_or, output_depth_mask).astype("uint8")
        valid_glass_mask = np.where((output_color_and > 0), 0, valid_glass_area)
        valid_glass_mask = np.where((valid_glass_mask > 0), 128, zero_mask).astype("uint8")

        # area where exist color mask and depth hole is glass area
        depth_hole_mask = np.where((depth == 0), 255, zero_mask).astype("uint8")

        
        
        # 0 64 128 192 256
        glass_mask = np.where((output_depth_mask > 0), 64, zero_mask)
        glass_mask = np.where((output_color_or > 0), 128, glass_mask)
        
        or_glass_hole = cv2.bitwise_and(output_color_or, depth_hole_mask).astype("uint8")
        glass_mask = np.where((or_glass_hole > 0), 192, glass_mask)
        
        and_glass_hole = cv2.bitwise_and(output_color_and, depth_hole_mask).astype("uint8")
        glass_mask = np.where((and_glass_hole > 0), 255, glass_mask)
        
        if(np.max(glass_mask) > 0):
            cv2.imwrite(os.path.join(destination_dir, filename.split('.')[0] + '_masked.png'), glass_mask)
            print(f'Image {filename}  have glass area')
        else:
            cv2.imwrite(os.path.join(destination_dir, filename.split('.')[0] + '_masked.png'), zero_mask)
            print(f'Image {filename} dont have glass area')
        
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()        