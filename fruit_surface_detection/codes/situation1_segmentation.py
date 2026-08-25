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



def kmeans_exact(pixels, k=3, max_iters=10):

    np.random.seed(42)

    indices = np.random.choice(pixels.shape[0], k, replace=False)

    centers = pixels[indices]

    

    labels = np.zeros(pixels.shape[0], dtype=np.int32)

    

    for _ in range(max_iters):

        distances = np.linalg.norm(pixels[:, None] - centers, axis=2)

        labels = np.argmin(distances, axis=1)

        new_centers = np.array([pixels[labels == i].mean(axis=0) if np.sum(labels == i) > 0 else centers[i] for i in range(k)])

        if np.allclose(centers, new_centers):

            break

        centers = new_centers

        

    return labels, np.float32(centers)







def color_thresholding(img):

    

       

    start_time = time.perf_counter()

    

                                            

    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    

                       

    hsv = bgr_to_hsi_exact(blurred)

    

                                                                                                  

                                                                                                           

    s_channel = hsv[:, :, 1]

    _, binary_mask = otsu_thresholding_exact(s_channel)

    

                              

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    

                                                         

    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_contour = None

    if contours:

        final_contour = max(contours, key=cv2.contourArea)

        

    proc_time = time.perf_counter() - start_time

    

    return cleaned_mask, final_contour, proc_time



def lab_kmeans(img, k=3):

    

       

    start_time = time.perf_counter()

    

                      

    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    

                       

    lab = bgr_to_lab_exact(blurred)

    

                            

    pixels = lab.reshape((-1, 3))

    pixels = np.float32(pixels)

    

                

    labels, centers = kmeans_exact(pixels, k=k, max_iters=10)

    

                                                                                                        

                                                                                                     

    centers_lab = np.uint8([centers])

                                                                      

    color_mag = np.sum((centers[:, 1:] - 128)**2, axis=1)

    fruit_cluster_idx = np.argmax(color_mag)

    

    labels_2d = labels.flatten()

    binary_mask = np.zeros(labels_2d.shape, dtype=np.uint8)

    binary_mask[labels_2d == fruit_cluster_idx] = 255

    binary_mask = binary_mask.reshape(img.shape[:2])

    

                              

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    

                                                         

    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_contour = None

    if contours:

        final_contour = max(contours, key=cv2.contourArea)

        

    proc_time = time.perf_counter() - start_time

    

    return cleaned_mask, final_contour, proc_time

