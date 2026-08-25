import numpy as np



def bgr_to_hsi_thesis(image):

    




       

    bgr = image.astype(np.float32) / 255.0

    B, G, R = bgr[:,:,0], bgr[:,:,1], bgr[:,:,2]

    

    V = (R + G + B) / 3.0

    

    min_rgb = np.min(bgr, axis=2)

    S = 1.0 - (min_rgb / (V + 1e-6))

    

                                                         

                                                                    

    num = np.sqrt(3) * (G - B)                                                                                                                                                                            

    num = 3.0 * (G - B)

    den = (R - B) + (R - B)

    

    H = np.tan(num / (den + 1e-6))

    

                                                                                         

    H_norm = ((H - np.min(H)) / (np.max(H) - np.min(H) + 1e-6)) * 179.0

    S_norm = S * 255.0

    V_norm = V * 255.0

    

    return np.stack([H_norm, S_norm, V_norm], axis=2).astype(np.float32)



def bgr_to_lab_thesis(image):

    




       

    bgr = image.astype(np.float32) / 255.0

    B, G, R = bgr[:,:,0], bgr[:,:,1], bgr[:,:,2]

    

                                 

    X = R * 0.412453 + G * 0.357580 + B * 0.180423

    Y = R * 0.212671 + G * 0.715160 + B * 0.072169

    Z = R * 0.019334 + G * 0.119193 + B * 0.950227

    

    Xn, Yn, Zn = 0.950456, 1.0, 1.088754

    

    def f(t):

        res = np.zeros_like(t)

        mask = t > 0.008856

        res[mask] = np.power(t[mask], 1.0/3.0)

        res[~mask] = (7.787 * t[~mask]) + (16.0 / 116.0)

        return res

        

    L = np.zeros_like(Y)

    y_ratio = Y / Yn

    mask = y_ratio > 0.008856

    L[mask] = 116.0 * np.power(y_ratio[mask], 1.0/3.0) - 16.0

    L[~mask] = 903.3 * y_ratio[~mask]

    

    a = 500.0 * (f(X/Xn) - f(Y/Yn))

    b = 200.0 * (f(Y/Yn) - f(Z/Zn))

    

    return np.stack([L, a, b], axis=2).astype(np.float32)

