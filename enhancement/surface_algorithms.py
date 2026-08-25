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





def lab_kmeans(img, k=3):

    start_time = time.perf_counter()

    

                           

    fg_mask = extract_fruit_foreground(img)

    

                                          

    blurred = cv2.GaussianBlur(img, (7, 7), 0)

    

                                                                   

    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)

    

                                            

    lab_masked = cv2.bitwise_and(lab, lab, mask=fg_mask)

    

    pixel_values = lab_masked.reshape((-1, 3))

    pixel_values = np.float32(pixel_values)

    

                                                                                             

    non_zero_indices = np.any(pixel_values != [0,0,0], axis=1)

    non_zero_pixels = pixel_values[non_zero_indices]

    

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    _, labels, (centers) = cv2.kmeans(non_zero_pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    

    centers = np.uint8(centers)

    labels = labels.flatten()

    

    segmented_non_zero = centers[labels]

    

                       

    segmented_data = np.zeros_like(pixel_values)

    segmented_data[non_zero_indices] = segmented_non_zero

    segmented_image = np.uint8(segmented_data.reshape(lab.shape))

    

                                                           

    bgr_segmented = cv2.cvtColor(segmented_image, cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(bgr_segmented, cv2.COLOR_BGR2GRAY)

    gray = gray.astype('uint8')

    

                            

    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    

                               

    mask[fg_mask == 0] = 0

    

    proc_time = time.perf_counter() - start_time

    return mask, None, proc_time



def color_thresholding(img):

    start_time = time.perf_counter()

    

    fg_mask = extract_fruit_foreground(img)

    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    

    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

                                                                 

    lower_bound = np.array([0, 50, 20])

    upper_bound = np.array([30, 255, 150])

    

    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    mask[fg_mask == 0] = 0

    

    kernel = np.ones((3,3), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea) if contours else None

    

    proc_time = time.perf_counter() - start_time

    return mask, largest_contour, proc_time



def morphological_watershed(img):

    start_time = time.perf_counter()



                           

    fg_mask = extract_fruit_foreground(img)

    

    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    

                                               

    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    

                                                             

    binary = cv2.bitwise_and(binary_mask, binary_mask, mask=fg_mask)



    kernel = np.ones((3,3), np.uint8)

    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)



    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    _, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)



    unknown = cv2.subtract(sure_bg, sure_fg)



    _, markers = cv2.connectedComponents(sure_fg)

    markers = markers + 1

    

                                                                                      

    markers[fg_mask == 0] = 1 

    markers[unknown == 255] = 0



    img_copy = img.copy()

    cv2.watershed(img_copy, markers)



    boundary_mask = np.zeros_like(gray)

    boundary_mask[markers == -1] = 255

                                                     

    boundary_mask[fg_mask == 0] = 0 



    raw_contours, _ = cv2.findContours(boundary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = [c for c in raw_contours if cv2.arcLength(c, False) > 15]



    filtered_boundary = np.zeros_like(boundary_mask)

    cv2.drawContours(filtered_boundary, contours, -1, 255, 1)



    proc_time = time.perf_counter() - start_time

    return filtered_boundary, contours, proc_time



def canny_edge_detection(img):

    start_time = time.perf_counter()

    

    fg_mask = extract_fruit_foreground(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    

    v = np.median(blurred[fg_mask > 0])                                     

    sigma = 0.33

    lower = int(max(0, (1.0 - sigma) * v))

    upper = int(min(255, (1.0 + sigma) * v))

    

    edges = cv2.Canny(blurred, lower, upper)

                             

    edges[fg_mask == 0] = 0

    

    raw_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = [c for c in raw_contours if cv2.contourArea(c) > 5 or cv2.arcLength(c, False) > 10]

    

    filtered_edges = np.zeros_like(edges)

    cv2.drawContours(filtered_edges, contours, -1, 255, 1)



    proc_time = time.perf_counter() - start_time

    return filtered_edges, contours, proc_time



from fruit_surface_detection.codes.situation3_inspection import run_active_contour, global_shape_descriptors

