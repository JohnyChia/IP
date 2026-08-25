import cv2

import numpy as np

import time

from skimage.segmentation import active_contour

from skimage.filters import gaussian



def global_shape_descriptors_exact(contour):

    S_c = cv2.contourArea(contour)

    if S_c == 0: return 0.0

    

    (x,y), radius = cv2.minEnclosingCircle(contour)

    S_e = np.pi * (radius ** 2)

    

    if S_e == 0: return 0.0

    return S_c / S_e





def global_shape_descriptors(img, contour):

    


       

    start_time = time.perf_counter()

    

    metrics = {}

    if hasattr(contour, 'ndim') and contour.ndim == 3 and contour.shape[0] >= 3:

        area = cv2.contourArea(contour)

        perimeter = cv2.arcLength(contour, True)

        

                                                               

        circularity = global_shape_descriptors_exact(contour)

        

                          

        hull = cv2.convexHull(contour)

        hull_area = cv2.contourArea(hull)

        

                                            

        solidity = float(area) / hull_area if hull_area > 0 else 0

        

                                     

        x, y, w, h = cv2.boundingRect(contour)

        aspect_ratio = float(w) / h if h > 0 else 0

        

        metrics = {

            'area': area,

            'perimeter': perimeter,

            'circularity': circularity,

            'hull_area': hull_area,

            'solidity': solidity,

            'aspect_ratio': aspect_ratio

        }

        

    proc_time = time.perf_counter() - start_time

    return metrics, proc_time



def run_active_contour(img, initial_contour):

    

       

    start_time = time.perf_counter()

    

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    

                                                                                               

    scale = 1.0

    h, w = gray.shape

    if w > 500 or h > 500:

        scale = 500.0 / max(w, h)

        gray = cv2.resize(gray, (int(w * scale), int(h * scale)))

    

                                                          

    smoothed = gaussian(gray, sigma=3, preserve_range=False)

    

    if initial_contour is None or len(initial_contour) < 3:

        h_s, w_s = gray.shape

        s = np.linspace(0, 2*np.pi, 100)

        r = h_s/2 + (h_s/4)*np.sin(s)

        c = w_s/2 + (w_s/4)*np.cos(s)

        init = np.array([r, c]).T

    else:

                                    

        init_scaled = initial_contour[:, 0, :] * scale

        init = init_scaled[:, ::-1]               

        

                                

                                                             

    snake = active_contour(smoothed, init, alpha=0.015, beta=10, gamma=0.001, max_num_iter=500)

    

                                                                                   

    final_contour_cv = np.zeros((len(snake), 1, 2), dtype=np.int32)

    final_contour_cv[:, 0, 0] = snake[:, 1] / scale

    final_contour_cv[:, 0, 1] = snake[:, 0] / scale

    

    proc_time = time.perf_counter() - start_time

    return final_contour_cv, proc_time

