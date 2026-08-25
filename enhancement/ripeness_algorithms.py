import cv2

import numpy as np

import time



def extract_fruit_foreground(img):

    

       

    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1,65), np.float64)

    fgdModel = np.zeros((1,65), np.float64)

    

    h, w = img.shape[:2]

    margin_x = int(w * 0.05)

    margin_y = int(h * 0.05)

    rect = (margin_x, margin_y, w - 2*margin_x, h - 2*margin_y)

    

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')

    

    kernel = np.ones((5,5), np.uint8)

    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=2)

    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel, iterations=1)

    

    return mask2 * 255



def segment_roi_enhanced(roi_img):

    


       

                               

    mask = extract_fruit_foreground(roi_img)

    

                                                                  

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    

    final_mask = np.zeros_like(mask)

    if contours:

                                                          

                                                              

        for i, contour in enumerate(contours):

                                                                                

            if cv2.contourArea(contour) > (roi_img.shape[0] * roi_img.shape[1] * 0.01):

                cv2.drawContours(final_mask, [contour], -1, 255, thickness=cv2.FILLED)

                

    return final_mask



from fruit_ripeness.codes.ripeness_assessment import (

    extract_hsv_features_roi, 

    extract_lab_features_roi, 

    classify_ripeness_hsv, 

    classify_ripeness_lab

)

