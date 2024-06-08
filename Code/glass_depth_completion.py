# -*- coding: utf-8 -*-
"""
Created on Tue May 16 18:16:02 2023

@author: Jia
https://stackoverflow.com/questions/69004202/two-color-linear-gradient-positioning-with-pil-python
使用 Python、NumPy 生成漸變圖像
https://note.nkmk.me/en/python-numpy-generate-gradation-image/
使用enlarge_mask找玻璃周圍的深度值
find_10_percent_indices: 找出遠近各前10%的深度值座標
find_center_of_mass: 找出最近處及最遠處的質心座標
linear_gradient_factor: 找出線性漸層係數
draw_gradient_points: 將深度圖塗上漸層
"""
import os
import glob
import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import stats

def crop_image_border(image, border_size):
    # Get image dimensions
    height, width = image.shape[:2]

    # Calculate the crop region
    crop_top = border_size
    crop_bottom = height - border_size
    crop_left = border_size
    crop_right = width - border_size

    # Crop the image
    cropped_image = image[crop_top:crop_bottom, crop_left:crop_right]

    return cropped_image

def enlarge_mask(image, contours, idx, thickness=2):
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
    for i in range(iterations):
        # contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)    
        draw_contour = cv2.drawContours(mask, contours, idx, (255,255,255), -1)
        draw_contour = cv2.drawContours(draw_contour, contours, idx, (255,255,255), thickness) # enlarge mask
    return draw_contour
# #======================================================================
# To create a linear gradient between two colors with a customizable vector
# from (x0, y0) to (x1, y1) in Python
# #======================================================================

def linear_gradient_factor(x0, y0, x1, y1, width, height):
    x, y = np.meshgrid(np.arange(width), np.arange(height), sparse=True)
    interpolation_factor = ((x - x0) * (x1 - x0) + (y - y0) * (y1 - y0)) / ((x1 - x0)**2 + (y1 - y0)**2)
    # #norm
    # interpolation_factor_norm = (interpolation_factor - np.min(interpolation_factor)) / (np.max(interpolation_factor) - np.min(interpolation_factor))
    
    return interpolation_factor # interpolation_factor_norm

def draw_gradient_points(width, height, factor, color1, color2):
    img = np.zeros((height,width), dtype=np.uint16)
    for y in range(factor.shape[0]):
            for x in range(factor.shape[1]):
                if color1 != color2:
                    interpolated_color = int((1 - factor[y][x]) * color1 + factor[y][x] * color2)
                else:
                    interpolated_color = int(color1)
                img[y][x] = np.array(interpolated_color).astype('uint16')
    return img.astype(np.uint16)

def generate_quadratic_gradient(color1, color2, x0, y0, x1, y1, width, height):
    img = np.zeros((height,width), dtype=np.uint16)
    
    for y in range(height):
        for x in range(width):
            # Calculate the interpolation factor based on the position along the vector
            distance = (x - x0) * (x1 - x0) + (y - y0) * (y1 - y0)
            interpolation_factor = distance / ((x1 - x0)**2 + (y1 - y0)**2)
            interpolation_factor = min(max(interpolation_factor, 0), 1)  # Clamp the factor between 0 and 1
            
            # Interpolate between color1 and color2 using quadratic easing function
            easing_factor = interpolation_factor * (2 - interpolation_factor)
            interpolated_color = (int((1 - easing_factor) * color1 + easing_factor * color2))
            img[y][x] = interpolated_color
            
    cv2.imshow('quadratic Gradient', img.astype(np.uint16)) # 16 bit depth
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img.astype(np.uint16)
# #======================================================================

def find_10_percent_indices(arr):
    # Flatten the 2D array into a 1D array
    flattened = arr.flatten()

    # Calculate the threshold value for top 10% and bottom 10%
    non_zero_values = flattened[flattened != 0]
    top_threshold = np.percentile(non_zero_values, 85)
    t_outlier_threshold = np.percentile(non_zero_values, 95)
    bottom_threshold = np.percentile(non_zero_values, 15)
    b_outlier_threshold = np.percentile(non_zero_values,5)

    # Find indices of values exceeding the threshold
    top_indices = np.argwhere((arr >= top_threshold) & (arr <= t_outlier_threshold) & (arr != 0))
    bottom_indices = np.argwhere((arr <= bottom_threshold) & (arr >= b_outlier_threshold) & (arr != 0))

    # Sort indices based on the corresponding values in descending order
    top_sorted_indices = top_indices[np.argsort(arr[top_indices[:, 0], top_indices[:, 1]])[::-1]]
    bottom_sorted_indices = bottom_indices[np.argsort(arr[bottom_indices[:, 0], bottom_indices[:, 1]])[::-1]]

    # Extract the top 10% indices
    top_10_percent_indices = top_sorted_indices[:int(len(top_sorted_indices) * 0.1)]
    bottom_10_percent_indices = bottom_sorted_indices[:int(len(bottom_sorted_indices) * 0.1)]

    return top_sorted_indices, bottom_sorted_indices

def find_custom_percent_indices(arr, coefficient):
    # Flatten the 2D array into a 1D array
    # coefficient: 1,2,3
    flattened = arr.flatten()
    top_start = 90 - (coefficient*10)
    bottom_start = 10 + (coefficient*10)
    if coefficient > 4:
        top_start = 50
        bottom_start = 50
        print("gradient direction wrong")
    # Calculate the threshold value for top 10% and bottom 10%
    non_zero_values = flattened[flattened != 0]
    top_threshold = np.percentile(non_zero_values, top_start)
    t_outlier_threshold = np.percentile(non_zero_values, 95)
    bottom_threshold = np.percentile(non_zero_values, bottom_start)
    b_outlier_threshold = np.percentile(non_zero_values, 5)

    # Find indices of values exceeding the threshold
    top_indices = np.argwhere((arr >= top_threshold) & (arr <= t_outlier_threshold) & (arr != 0))
    bottom_indices = np.argwhere((arr <= bottom_threshold) & (arr >= b_outlier_threshold) & (arr != 0))

    # Sort indices based on the corresponding values in descending order
    top_sorted_indices = top_indices[np.argsort(arr[top_indices[:, 0], top_indices[:, 1]])[::-1]]
    bottom_sorted_indices = bottom_indices[np.argsort(arr[bottom_indices[:, 0], bottom_indices[:, 1]])[::-1]]

    # Extract the top 10% indices
    top_10_percent_indices = top_sorted_indices[:int(len(top_sorted_indices) * 0.1)]
    bottom_10_percent_indices = bottom_sorted_indices[:int(len(bottom_sorted_indices) * 0.1)]

    return top_sorted_indices, bottom_sorted_indices

def find_center_of_mass(coordinates, weights):
    # Calculate the weighted average of coordinates
    center_y = np.average(coordinates[:, 0], weights=weights)
    center_x = np.average(coordinates[:, 1], weights=weights)

    return center_y, center_x

def find_factor(interested_depth):
        
    top_10_percent_indices, bottom_10_percent_indices = find_10_percent_indices(interested_depth)    
    top_values = interested_depth[top_10_percent_indices[:, 0], top_10_percent_indices[:, 1]]
    bottom_values = interested_depth[bottom_10_percent_indices[:, 0], bottom_10_percent_indices[:, 1]]

    top_y, top_x = find_center_of_mass(top_10_percent_indices, top_values)
    bottom_y, bottom_x = find_center_of_mass(bottom_10_percent_indices, bottom_values)

    x0 = int(bottom_x)
    y0 = int(bottom_y)
    x1 = int(top_x)
    y1 = int(top_y)
    return x0, y0, x1, y1

def find_factor_min_max(arr):
    flattened = arr.flatten()
    non_zero_values = flattened[flattened != 0]
    t_outlier_threshold = np.percentile(non_zero_values, 95)
    b_outlier_threshold = np.percentile(non_zero_values, 5)
    indices = np.argwhere((arr >= b_outlier_threshold) & (arr <= t_outlier_threshold) & (arr != 0))
    
    max_value = np.argmax(indices)
    max_index_2d = np.unravel_index(max_value, arr.shape)
    # print("max_index_2d", max_index_2d)
    top_y, top_x = max_index_2d    
    
    min_value = np.argmin(indices)
    min_index_2d = np.unravel_index(min_value, arr.shape)
    bottom_y, bottom_x = min_index_2d
    # print("min_index_2d", min_index_2d)
    
    return bottom_x, bottom_y, top_x, top_y

def find_factor_bbox(image, contour):
    
    # rect = cv2.minAreaRect(contour)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # print("box:", box.shape)
    # print("box:", box)
    
    c = contour[1]
    c2 = c[:-1]
    # compute the extreme points of the contour
    leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
    rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
    topmost = tuple(contour[contour[:, :, 1].argmin()][0])
    bottommost = tuple(contour[contour[:, :, 1].argmax()][0])
    points = [leftmost, rightmost, topmost, bottommost]
    # print("points:", points)
    values = [image[leftmost], image[rightmost], 
              image[topmost], image[bottommost]]
    
    # points = [(box[0][0], box[0][1]), (box[1][0], box[1][1]), 
    #           (box[2][0], box[2][1]), (box[3][0], box[3][1])] #image[y:y+h, x:x+w]
    # values = [image[box[0][0]][box[0][1]], image[box[1][0]][box[1][1]], 
    #           image[box[2][0]][box[2][1]], image[box[3][0]][box[3][1]]]
    # print("values:", values)
    
    # Combine points and values using zip
    combined = list(zip(points, values))
    
    # Sort the combined list based on values in descending order
    sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
    
    # Extract the top 2 and bottom 2 points and values
    top_2 = sorted_combined[:2]
    bottom_2 = sorted_combined[-2:]
    
    # Separate points and values from the top 2 and bottom 2
    top_2_points, top_2_values = zip(*top_2)
    bottom_2_points, bottom_2_values = zip(*bottom_2)
    # print("top_2_points:", np.array(top_2_points))
    # print("top_2_values:", top_2_values)
    
    top_y, top_x = find_center_of_mass(np.array(top_2_points), top_2_values)
    bottom_y, bottom_x = find_center_of_mass(np.array(bottom_2_points), bottom_2_values)

    x0 = int(bottom_x)
    y0 = int(bottom_y)
    x1 = int(top_x)
    y1 = int(top_y)

    return x0, y0, x1, y1

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
    img_blur = cv2.GaussianBlur(img_gray, (5,5), 0)
    img_blur = img_blur.astype(np.uint8)
    # cv2.imshow('img_blur', img_blur)
    # Canny Edge Detection
    edges = cv2.Canny(img_blur, 150, 250) # Canny Edge Detection
    # cv2.imshow('edges', edges)
    check_mask = np.where(mask, edges, 0) # edges is 255
    check_val = np.max(check_mask)
    high_contrast = True
    if check_val > 0:
        high_contrast = True
        print("The mask region has high contrast")
    else:
        high_contrast = False
        # print("The mask region has adjacent pixels within the threshold distance.")
    return high_contrast

def create_gradient_map(image, interested_area, mask, is_contrast):    
    
    # check if gradient_map is high contrast
    is_contrast = True
    coefficient = 0
             
    x0, y0, x1, y1 = find_factor(interested_area)

    color1 = image[int(y0)][int(x0)]
    color2 = image[int(y1)][int(x1)]
    # print("color", color1, color2)
    
    if (color1 == 0) or (color2 == 0):
        x0, y0, x1, y1 = find_factor_min_max(interested_area)
        color1 = image[int(y0)][int(x0)]
        color2 = image[int(y1)][int(x1)]
        # print("color or", color1, color2)    
        
    height, width = image.shape
    interpolation_factor = linear_gradient_factor(int(x0), int(y0), int(x1), int(y1), width, height)
    gradient_map = draw_gradient_points(width, height, interpolation_factor, color1, color2)
    
    is_contrast = check_high_contrast(gradient_map, mask)   
    while(is_contrast):
        coefficient = coefficient + 1
        color = int((color1 + color2) / 2)
        offset = int((color1 + color2) / ((1 + coefficient)*2))
        color1 = color - offset
        color2 = color + offset
        gradient_map = draw_gradient_points(width, height, interpolation_factor, color1, color2)
        is_contrast = check_high_contrast(gradient_map, mask)   
        # print("color", color1, color2)
        # print("======coefficient:", coefficient)
    
    # thresh = color2 - color1 # max depth - min depth, must >= 0
    return gradient_map


if __name__ == "__main__":
    # Specify the destination directory where the files will be copied
    raw_dir = r"G:/matterport3D/M3D_mask/mask_20/raw_20" # raw input
    depth_dir = r"G:/matterport3D/M3D_mask/mask_20/raw_clean_20" # preprocessing depth image
    mask_dir = r"G:/matterport3D/M3D_mask/mask_20/grabCut_20_01_80"
    destination_dir = r"G:/matterport3D/M3D_mask/mask_20/grabCut_20_no_mask"
    
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
        
    ################################################
    #==========read images from directory==========#
    ################################################
    raw_file_list = []
    depth_file_list = []
    EXT_RAW_IMG = ['.png']
    EXT_DEPTH_IMG = ['.png']
    for ext in EXT_RAW_IMG:
        raw_file_list += (sorted(glob.glob(os.path.join(raw_dir, '*' + ext))))
    for ext in EXT_DEPTH_IMG:
        depth_file_list += (sorted(glob.glob(os.path.join(depth_dir, '*' + ext))))
        
    for i in range(len(raw_file_list)):
        filename = os.path.basename(raw_file_list[i])
        depth_file = depth_file_list[i]
        clean_depth = cv2.imread(depth_file_list[i], cv2.IMREAD_UNCHANGED)
        clean_depth = cv2.resize(clean_depth, (320, 256), interpolation=cv2.INTER_AREA)
        clean_depth[np.isnan(clean_depth)] = 0
        clean_depth[np.isinf(clean_depth)] = 0
        inpaint_mask = np.where(clean_depth, 0, 255).astype(np.uint8) #補沒值的地方
        inpaint_dst = clean_depth.copy() 
        inpaint_dst = cv2.inpaint(inpaint_dst, inpaint_mask, 25, cv2.INPAINT_TELEA)
        # inpaint_dst = cv2.GaussianBlur(inpaint_dst, (9,9), 0)
        print("clean_depth.shape",clean_depth.shape)

        raw_depth = cv2.imread(raw_file_list[i], cv2.IMREAD_UNCHANGED)
        fill_image = raw_depth.copy()
        file_path = filename.replace(".png", "_masked.png").replace("_d", "_i").replace("resize_i", "resize_d")
        mask_file =  os.path.join(mask_dir, file_path)
        valid_mask = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
        valid_mask = cv2.resize(valid_mask, (320, 256), interpolation=cv2.INTER_AREA)
        input_mask = np.where((valid_mask > 0), 255, 0).astype(np.uint8) # white area in mask
        
        # # fill depth value where raw_depth == 0
        # input_mask = np.where((raw_depth == 0), 255, 0).astype(np.uint8) 
        
        input_contours, hierarchy = cv2.findContours(input_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)        
        for i in range(len(input_contours)):  
            large_mask = enlarge_mask(input_mask, input_contours, i)
            
            o_mask = cv2.bitwise_and(input_mask,large_mask)
            interested_mask = cv2.bitwise_xor(o_mask,large_mask) # 0 and 255
            # fill_mask = cv2.drawContours(interested_mask, input_contours, i, (255,255,255), -1)
            # fill_mask = cv2.bitwise_and(interested_mask,interested_mask)            
            deep = np.median(inpaint_dst[interested_mask>0])

            if (deep > 0) and (np.count_nonzero(interested_mask > 0) > 20): # valid interested_mask
                # cv2.imshow('interested_mask', interested_mask)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                valid = (interested_mask > 0) # & (inpaint_dst > (sum_mode*0.9)) # (inpaint_dst > sum_mode
                interested_depth = np.where(valid, inpaint_dst, 0)

                
                gradient_map = create_gradient_map(inpaint_dst, interested_depth, large_mask, is_contrast=False)
                
                gradient_map = np.asarray(gradient_map)        
                # cv2.imwrite(os.path.join(destination_dir, filename.split('.')[0] + '_gradient_map_' + str(i) + '.png'), gradient_map.astype(np.uint16))
                fill_hole = np.where(large_mask, gradient_map, fill_image).astype(np.uint16) #gt map
                # print("gradient_map mean:", np.min(gradient_map[o_mask>0]))
                fill_image = np.where((fill_hole>0), fill_hole, fill_image).astype(np.uint16)   
                
  
        cv2.imwrite(os.path.join(destination_dir, filename.split('.')[0] + '_masked.png'), fill_image.astype(np.uint16))
        
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()