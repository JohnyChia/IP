import cv2

import numpy as np

import time

from fruit_ripeness.codes.thesis_formulas import bgr_to_hsi_thesis, bgr_to_lab_thesis



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



def bgr_to_hsi_exact(image):

    bgr = image.astype(np.float32) / 255.0

    B, G, R = bgr[:,:,0], bgr[:,:,1], bgr[:,:,2]

    

    V = (R + G + B) / 3.0

    

    min_rgb = np.minimum(np.minimum(R, G), B)

    S = np.zeros_like(V)

    non_zero_v = V > 0

    S[non_zero_v] = 1.0 - min_rgb[non_zero_v] / V[non_zero_v]

    

    numerator = 3.0 * (G - B)

    denominator = (R - B) + (R - B)

    

    H_val = np.zeros_like(V)

    valid = denominator != 0

    H_val[valid] = np.tan(numerator[valid] / denominator[valid])

    

    H = np.clip(H_val * 179, 0, 179).astype(np.uint8)

    S = (S * 255).astype(np.uint8)

    V = (V * 255).astype(np.uint8)

    

    return np.stack([H, S, V], axis=2)



def bgr_to_lab_exact(image):

    bgr = image.astype(np.float32) / 255.0

    B, G, R = bgr[:,:,0], bgr[:,:,1], bgr[:,:,2]

    

    def inv_srgb(c):

        mask = c > 0.04045

        res = np.zeros_like(c)

        res[mask] = np.power((c[mask] + 0.055) / 1.055, 2.4)

        res[~mask] = c[~mask] / 12.92

        return res * 100.0

        

    R_lin, G_lin, B_lin = inv_srgb(R), inv_srgb(G), inv_srgb(B)

    X = R_lin * 0.412453 + G_lin * 0.357580 + B_lin * 0.180423

    Y = R_lin * 0.212671 + G_lin * 0.715160 + B_lin * 0.072169

    Z = R_lin * 0.019334 + G_lin * 0.119193 + B_lin * 0.950227

    

    Xn, Yn, Zn = 95.047, 100.000, 108.883

    

    def f(t):

        mask = t > 0.008856

        res = np.zeros_like(t)

        res[mask] = np.power(t[mask], 1.0/3.0)

        res[~mask] = (7.787 * t[~mask]) + (16.0 / 116.0)

        return res

        

    fx, fy, fz = f(X/Xn), f(Y/Yn), f(Z/Zn)

    

    L = np.zeros_like(Y)

    y_mask = (Y/Yn) > 0.008856

    L[y_mask] = 116.0 * np.power(Y[y_mask]/Yn, 1.0/3.0) - 16.0

    L[~y_mask] = 903.3 * (Y[~y_mask] / Yn)

    

    a = 500.0 * (fx - fy)

    b = 200.0 * (fy - fz)

    

    L = (L * 255.0 / 100.0).astype(np.uint8)

    a = (a + 128.0).clip(0, 255).astype(np.uint8)

    b = (b + 128.0).clip(0, 255).astype(np.uint8)

    

    return np.stack([L, a, b], axis=2)





def segment_roi(roi_img):

    



       

    if roi_img.size == 0:

        return np.zeros((1, 1), dtype=np.uint8)

        

    hsv = bgr_to_hsi_thesis(roi_img)

    s_channel = hsv[:, :, 1]

    

                                     

    _, mask = otsu_thresholding_exact(s_channel)

    

                                       

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    

                          

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    

    final_mask = np.zeros_like(mask)

    if contours:

        largest_contour = max(contours, key=cv2.contourArea)

                                                                      

        if cv2.contourArea(largest_contour) > (roi_img.shape[0] * roi_img.shape[1] * 0.05):

            cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

            

    return final_mask



def extract_hsv_features_roi(roi_img, foreground_mask, situation=1):

    start_time = time.perf_counter()

    if situation == 1:

                                                  

        hsv = bgr_to_hsi_thesis(roi_img)

    else:

                                                                           

        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

        

    fg_pixels = hsv[foreground_mask > 0]

    if fg_pixels.size > 0:

        mean_h, mean_s, mean_v = fg_pixels.mean(axis=0)

    else:

        mean_h, mean_s, mean_v = 0.0, 0.0, 0.0

    time_taken = time.perf_counter() - start_time

    return mean_h, mean_s, mean_v, time_taken



def extract_lab_features_roi(roi_img, foreground_mask, situation=1):

    start_time = time.perf_counter()

    if situation == 1:

                                                  

        lab = bgr_to_lab_thesis(roi_img)

    else:

                                                                           

        lab = cv2.cvtColor(roi_img, cv2.COLOR_BGR2LAB)

        

    fg_pixels = lab[foreground_mask > 0]

    if fg_pixels.size > 0:

        mean_l, mean_a, mean_b = fg_pixels.mean(axis=0)

    else:

        mean_l, mean_a, mean_b = 0.0, 0.0, 0.0

    time_taken = time.perf_counter() - start_time

    return mean_l, mean_a, mean_b, time_taken



def classify_ripeness_hsv(mean_h):

    






       

    if mean_h < 20 or mean_h > 120:

        return "Overripe"

    elif 20 <= mean_h <= 45:

        return "Ripe"

    else:

        return "Unripe"



def classify_ripeness_lab(mean_a):

    





       

    if mean_a < 110:

        return "Unripe"

    elif 110 <= mean_a <= 150:

        return "Ripe"

    else:

        return "Overripe"

