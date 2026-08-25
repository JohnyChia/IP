import cv2

import numpy as np

import time



def otsu_thresholding_exact(image):

    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))

    p = hist / image.size

    

    max_var = -1

    best_thresh = 0

    for T in range(256):

        p0_T = np.sum(p[:T+1])

        p1_T = np.sum(p[T+1:])

        if p0_T == 0 or p1_T == 0:

            continue

        mu0_T = np.sum(np.arange(T+1) * p[:T+1]) / p0_T

        mu1_T = np.sum(np.arange(T+1, 256) * p[T+1:]) / p1_T

        

        sigma_W_sq = p0_T * p1_T * (mu0_T - mu1_T)**2

        if sigma_W_sq > max_var:

            max_var = sigma_W_sq

            best_thresh = T

            

    binary_mask = np.zeros_like(image, dtype=np.uint8)

    binary_mask[image >= best_thresh] = 255

    return best_thresh, binary_mask



def canny_gradients_exact(img, sigma=1.0):

    if len(img.shape) == 3:

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        

    img = img.astype(np.float32)

    

    size = int(6 * sigma + 1)

    if size % 2 == 0: size += 1

    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))

    g = g / (2 * np.pi * sigma**2)

    

    smoothed = cv2.filter2D(img, -1, g)

    

    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)

    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    

    Ix = cv2.filter2D(smoothed, -1, Gx)

    Iy = cv2.filter2D(smoothed, -1, Gy)

    

    G_mag = np.sqrt(Ix**2 + Iy**2)

    theta = np.arctan2(Iy, Ix)

    

    return G_mag, theta





def canny_edge_detection(img):

    


       

    start_time = time.perf_counter()

    

                                                        

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    

                             

                                                                      

    v = np.median(blurred)

                                                

    sigma = 0.33

    lower = int(max(0, (1.0 - sigma) * v))

    upper = int(min(255, (1.0 + sigma) * v))

    

                                              

    if lower >= upper:

        lower = max(0, upper - 1)

        if lower == upper:                 

            upper = min(255, lower + 1)

            

                                               

    G_mag, theta = canny_gradients_exact(gray, sigma=1.0)

    

                                                                                  

    edges = np.zeros_like(gray, dtype=np.uint8)

    edges[G_mag > upper] = 255

    edges[G_mag < lower] = 0

                                                                                                      



    

                                         

    raw_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    

                                                                         

    contours = []

    for c in raw_contours:

        if cv2.arcLength(c, False) > 10 or cv2.contourArea(c) > 5:

            contours.append(c)



                                                                                                

    filtered_edges = np.zeros_like(edges)

    cv2.drawContours(filtered_edges, contours, -1, 255, 1)



    proc_time = time.perf_counter() - start_time

    return filtered_edges, contours, proc_time



def morphological_watershed(img):

    


       

    start_time = time.perf_counter()



                                      

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)



                                                                          

                                      

                                                                          

    _, binary_mask = otsu_thresholding_exact(blurred)

    binary = cv2.bitwise_not(binary_mask)



                                             

    kernel = np.ones((3,3), np.uint8)

    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)



                                                

    sure_bg = cv2.dilate(opening, kernel, iterations=3)



                                                          

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

                                                                         

    _, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)



                                

    unknown = cv2.subtract(sure_bg, sure_fg)



                                             

    _, markers = cv2.connectedComponents(sure_fg)

    

                                                                   

    markers = markers + 1

                                          

    markers[unknown == 255] = 0



                                              

    img_copy = img.copy()

    cv2.watershed(img_copy, markers)



                                                           

    boundary_mask = np.zeros_like(gray)

    boundary_mask[markers == -1] = 255

    

                                           

    raw_contours, _ = cv2.findContours(boundary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    

    contours = []

    for c in raw_contours:

        if cv2.arcLength(c, False) > 10 or cv2.contourArea(c) > 5:

            contours.append(c)

            

    filtered_boundary = np.zeros_like(boundary_mask)

    cv2.drawContours(filtered_boundary, contours, -1, 255, 1)



    proc_time = time.perf_counter() - start_time

    return filtered_boundary, contours, proc_time

