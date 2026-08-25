import os

import sys

import cv2

import numpy as np

import base64

from io import BytesIO

from PIL import Image

from flask import Flask, request, jsonify

from flask_cors import CORS

import time



                                          

sys.path.append(os.path.dirname(os.path.abspath(__file__)))



                                         

from fruit_surface_detection.codes.situation1_segmentation import lab_kmeans as orig_lab_kmeans, color_thresholding as orig_color_thresholding

from fruit_surface_detection.codes.situation2_boundary import morphological_watershed as orig_morphological_watershed, canny_edge_detection as orig_canny_edge_detection

from fruit_surface_detection.codes.situation3_inspection import run_active_contour as orig_run_active_contour, global_shape_descriptors as orig_global_shape_descriptors



                                           

from fruit_ripeness.codes.ripeness_assessment import (

    segment_roi as orig_segment_roi, 

    extract_hsv_features_roi as orig_extract_hsv_features_roi, 

    extract_lab_features_roi as orig_extract_lab_features_roi, 

    classify_ripeness_hsv as orig_classify_ripeness_hsv, 

    classify_ripeness_lab as orig_classify_ripeness_lab

)



                                         

from enhancement.surface_algorithms import (

    lab_kmeans as enh_lab_kmeans, 

    color_thresholding as enh_color_thresholding,

    morphological_watershed as enh_morphological_watershed, 

    canny_edge_detection as enh_canny_edge_detection,

    run_active_contour as enh_run_active_contour, 

    global_shape_descriptors as enh_global_shape_descriptors

)



                                           

from enhancement.ripeness_algorithms import (

    segment_roi_enhanced as enh_segment_roi, 

    extract_hsv_features_roi as enh_extract_hsv_features_roi, 

    extract_lab_features_roi as enh_extract_lab_features_roi, 

    classify_ripeness_hsv as enh_classify_ripeness_hsv, 

    classify_ripeness_lab as enh_classify_ripeness_lab

)



app = Flask(__name__)

CORS(app)                               



def cv2_to_base64(img):

    if img is None:

        return ""

                    

    _, buffer = cv2.imencode('.jpg', img)

                       

    b64_str = base64.b64encode(buffer).decode('utf-8')

    return f"data:image/jpeg;base64,{b64_str}"



def overlay_mask(img, mask, color=(0, 255, 0), alpha=0.4):

    overlay = img.copy()

    overlay[mask > 0] = color

    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)



def overlay_edges(img, edges, color=(0, 0, 255), thickness=2):

    res = img.copy()

    if thickness > 1:

                                                

        kernel = np.ones((thickness, thickness), np.uint8)

        edges = cv2.dilate(edges, kernel, iterations=1)

    res[edges > 0] = color

    return res



def is_fruit(img):

    



       

                                                        

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    s_channel = hsv[:, :, 1]

    _, mask = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:

        return False

        

    largest = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(largest)

    

    img_area = img.shape[0] * img.shape[1]

                                                                                                     

    if area < img_area * 0.05:

        return False

        

    return True



def process_image_frame(img):

                                                                                  

    max_dim = 600

    h, w = img.shape[:2]

    if max(h, w) > max_dim:

        scale = max_dim / max(h, w)

        img = cv2.resize(img, (int(w * scale), int(h * scale)))



                        

    if not is_fruit(img):

        return {

            "success": False,

            "error": "Not a valid fruit image. Please upload a clear image of a fruit."

        }



    results = {

        "success": True,

        "original_image": cv2_to_base64(img),

        "surface": {},

        "ripeness": {}

    }

    

                               

    

                                                             

              

    orig_mask_km, _, orig_time_km = orig_lab_kmeans(img.copy(), k=3)

    orig_mask_ct, _, orig_time_ct = orig_color_thresholding(img.copy())

              

    enh_mask_km, _, enh_time_km = enh_lab_kmeans(img.copy(), k=3)

    enh_mask_ct, _, enh_time_ct = enh_color_thresholding(img.copy())

    

    img_orig_km = overlay_mask(img.copy(), orig_mask_km, color=(255, 0, 0))

    img_orig_ct = overlay_mask(img.copy(), orig_mask_ct, color=(0, 255, 0))

    img_enh_km = overlay_mask(img.copy(), enh_mask_km, color=(255, 0, 0))

    img_enh_ct = overlay_mask(img.copy(), enh_mask_ct, color=(0, 255, 0))

    

    results["surface"]["sit1"] = {

        "alg1_name": "LAB K-Means",

        "alg2_name": "Color Thresholding",

        "orig_alg1_time": round(orig_time_km, 4),

        "orig_alg2_time": round(orig_time_ct, 4),

        "orig_alg1_img": cv2_to_base64(img_orig_km),

        "orig_alg2_img": cv2_to_base64(img_orig_ct),

        "enh_alg1_time": round(enh_time_km, 4),

        "enh_alg2_time": round(enh_time_ct, 4),

        "enh_alg1_img": cv2_to_base64(img_enh_km),

        "enh_alg2_img": cv2_to_base64(img_enh_ct)

    }

    

                                                            

              

    orig_mask_w, _, orig_time_w = orig_morphological_watershed(img.copy())

    orig_mask_c, _, orig_time_c = orig_canny_edge_detection(img.copy())

              

    enh_mask_w, _, enh_time_w = enh_morphological_watershed(img.copy())

    enh_mask_c, _, enh_time_c = enh_canny_edge_detection(img.copy())

    

    img_orig_w = overlay_edges(img.copy(), orig_mask_w, color=(0, 0, 255), thickness=2)

    img_orig_c = overlay_edges(img.copy(), orig_mask_c, color=(255, 0, 255), thickness=1)

    img_enh_w = overlay_edges(img.copy(), enh_mask_w, color=(0, 0, 255), thickness=2)

    img_enh_c = overlay_edges(img.copy(), enh_mask_c, color=(255, 0, 255), thickness=1)

    

    results["surface"]["sit2"] = {

        "alg1_name": "Morphological Watershed",

        "alg2_name": "Canny Edge Detection",

        "orig_alg1_time": round(orig_time_w, 4),

        "orig_alg2_time": round(orig_time_c, 4),

        "orig_alg1_img": cv2_to_base64(img_orig_w),

        "orig_alg2_img": cv2_to_base64(img_orig_c),

        "enh_alg1_time": round(enh_time_w, 4),

        "enh_alg2_time": round(enh_time_c, 4),

        "enh_alg1_img": cv2_to_base64(img_enh_w),

        "enh_alg2_img": cv2_to_base64(img_enh_c)

    }

    

                                                  

                                                                                     

    _, init_contour, _ = orig_color_thresholding(img.copy())

    _, enh_init_contour, _ = enh_color_thresholding(img.copy())

    

    orig_shape_metrics, orig_time_g = orig_global_shape_descriptors(img.copy(), init_contour)

    orig_final_contour_ac, orig_time_ac = orig_run_active_contour(img.copy(), init_contour)

    

    enh_shape_metrics, enh_time_g = enh_global_shape_descriptors(img.copy(), enh_init_contour)

    enh_final_contour_ac, enh_time_ac = enh_run_active_contour(img.copy(), enh_init_contour)

    

    img_orig_ac = img.copy()

    if orig_final_contour_ac is not None and len(orig_final_contour_ac) > 0:

        cv2.drawContours(img_orig_ac, [orig_final_contour_ac], 0, (0,255,0), 2)

    img_orig_init = img.copy()

    if init_contour is not None and len(init_contour) > 0:

        cv2.drawContours(img_orig_init, [init_contour], 0, (255,0,0), 2)



    img_enh_ac = img.copy()

    if enh_final_contour_ac is not None and len(enh_final_contour_ac) > 0:

        cv2.drawContours(img_enh_ac, [enh_final_contour_ac], 0, (0,255,0), 2)

    img_enh_init = img.copy()

    if enh_init_contour is not None and len(enh_init_contour) > 0:

        cv2.drawContours(img_enh_init, [enh_init_contour], 0, (255,0,0), 2)

        

    results["surface"]["sit3"] = {

        "alg1_name": "Active Contour",

        "alg2_name": "Global Shape Descriptors",

        "orig_alg1_time": round(orig_time_ac, 4),

        "orig_alg2_time": round(orig_time_g, 4),

        "orig_alg1_img": cv2_to_base64(img_orig_ac),

        "orig_alg2_img": cv2_to_base64(img_orig_init),

        "enh_alg1_time": round(enh_time_ac, 4),

        "enh_alg2_time": round(enh_time_g, 4),

        "enh_alg1_img": cv2_to_base64(img_enh_ac),

        "enh_alg2_img": cv2_to_base64(img_enh_init)

    }



                                 

                  

    orig_roi_mask = orig_segment_roi(img.copy())

                  

    enh_roi_mask = enh_segment_roi(img.copy())

    

                                        

    s1_orig_mean_h, s1_orig_mean_s, s1_orig_mean_v, _ = orig_extract_hsv_features_roi(img.copy(), orig_roi_mask, situation=1)

    s1_orig_class_hsv = orig_classify_ripeness_hsv(s1_orig_mean_h)

    

    s1_enh_mean_h, s1_enh_mean_s, s1_enh_mean_v, _ = orig_extract_hsv_features_roi(img.copy(), enh_roi_mask, situation=1)

    s1_enh_class_hsv = orig_classify_ripeness_hsv(s1_enh_mean_h)

    

    s1_orig_mean_l, s1_orig_mean_a, s1_orig_mean_b, _ = orig_extract_lab_features_roi(img.copy(), orig_roi_mask, situation=1)

    s1_orig_class_lab = orig_classify_ripeness_lab(s1_orig_mean_a)



    s1_enh_mean_l, s1_enh_mean_a, s1_enh_mean_b, _ = orig_extract_lab_features_roi(img.copy(), enh_roi_mask, situation=1)

    s1_enh_class_lab = orig_classify_ripeness_lab(s1_enh_mean_a)

    

                                                        

    s2_orig_mean_h, s2_orig_mean_s, s2_orig_mean_v, _ = orig_extract_hsv_features_roi(img.copy(), orig_roi_mask, situation=2)

    s2_orig_class_hsv = orig_classify_ripeness_hsv(s2_orig_mean_h)

    

    s2_enh_mean_h, s2_enh_mean_s, s2_enh_mean_v, _ = orig_extract_hsv_features_roi(img.copy(), enh_roi_mask, situation=2)

    s2_enh_class_hsv = orig_classify_ripeness_hsv(s2_enh_mean_h)

    

    s2_orig_mean_l, s2_orig_mean_a, s2_orig_mean_b, _ = orig_extract_lab_features_roi(img.copy(), orig_roi_mask, situation=2)

    s2_orig_class_lab = orig_classify_ripeness_lab(s2_orig_mean_a)



    s2_enh_mean_l, s2_enh_mean_a, s2_enh_mean_b, _ = orig_extract_lab_features_roi(img.copy(), enh_roi_mask, situation=2)

    s2_enh_class_lab = orig_classify_ripeness_lab(s2_enh_mean_a)

    

                                          

    orig_ripeness_img = img.copy()

    orig_ripeness_img[orig_roi_mask == 0] = [0,0,0]

    

    enh_ripeness_img = img.copy()

    enh_ripeness_img[enh_roi_mask == 0] = [0,0,0]

    

    results["ripeness"] = {

        "orig_ripeness_image": cv2_to_base64(orig_ripeness_img),

        "enh_ripeness_image": cv2_to_base64(enh_ripeness_img),

        "sit1": {

            "orig_class_hsv": s1_orig_class_hsv,

            "orig_class_lab": s1_orig_class_lab,

            "enh_class_hsv": s1_enh_class_hsv,

            "enh_class_lab": s1_enh_class_lab,

            "orig_hsv": {"h": float(s1_orig_mean_h), "s": float(s1_orig_mean_s), "v": float(s1_orig_mean_v)},

            "enh_hsv": {"h": float(s1_enh_mean_h), "s": float(s1_enh_mean_s), "v": float(s1_enh_mean_v)},

            "orig_lab": {"l": float(s1_orig_mean_l), "a": float(s1_orig_mean_a), "b": float(s1_orig_mean_b)},

            "enh_lab": {"l": float(s1_enh_mean_l), "a": float(s1_enh_mean_a), "b": float(s1_enh_mean_b)}

        },

        "sit2": {

            "orig_class_hsv": s2_orig_class_hsv,

            "orig_class_lab": s2_orig_class_lab,

            "enh_class_hsv": s2_enh_class_hsv,

            "enh_class_lab": s2_enh_class_lab,

            "orig_hsv": {"h": float(s2_orig_mean_h), "s": float(s2_orig_mean_s), "v": float(s2_orig_mean_v)},

            "enh_hsv": {"h": float(s2_enh_mean_h), "s": float(s2_enh_mean_s), "v": float(s2_enh_mean_v)},

            "orig_lab": {"l": float(s2_orig_mean_l), "a": float(s2_orig_mean_a), "b": float(s2_orig_mean_b)},

            "enh_lab": {"l": float(s2_enh_mean_l), "a": float(s2_enh_mean_a), "b": float(s2_enh_mean_b)}

        }

    }





                                    

                                           

    fg_ratio_orig_ct = float(np.sum(orig_mask_ct > 0) / orig_mask_ct.size)

    fg_ratio_orig_km = float(np.sum(orig_mask_km > 0) / orig_mask_km.size)

    fg_ratio_enh_ct = float(np.sum(enh_mask_ct > 0) / enh_mask_ct.size)

    fg_ratio_enh_km = float(np.sum(enh_mask_km > 0) / enh_mask_km.size)

    

                                          

    fg_ratio_orig_roi = float(np.sum(orig_roi_mask > 0) / orig_roi_mask.size)

    fg_ratio_enh_roi = float(np.sum(enh_roi_mask > 0) / enh_roi_mask.size)

    

                                          

    edge_density_orig_w = float(np.sum(orig_mask_w > 0) / orig_mask_w.size)

    edge_density_orig_c = float(np.sum(orig_mask_c > 0) / orig_mask_c.size)

    edge_density_enh_w = float(np.sum(enh_mask_w > 0) / enh_mask_w.size)

    edge_density_enh_c = float(np.sum(enh_mask_c > 0) / enh_mask_c.size)

    

    results["metrics"] = {

        "surface": {

            "sit1_orig_ct_fg": fg_ratio_orig_ct,

            "sit1_orig_km_fg": fg_ratio_orig_km,

            "sit1_enh_ct_fg": fg_ratio_enh_ct,

            "sit1_enh_km_fg": fg_ratio_enh_km,

            

            "sit2_orig_w_edge": edge_density_orig_w,

            "sit2_orig_c_edge": edge_density_orig_c,

            "sit2_enh_w_edge": edge_density_enh_w,

            "sit2_enh_c_edge": edge_density_enh_c,



            

            "sit3_orig_circularity": orig_shape_metrics.get("circularity", 0) if orig_shape_metrics else 0,

            "sit3_orig_solidity": orig_shape_metrics.get("solidity", 0) if orig_shape_metrics else 0,

            "sit3_orig_aspect_ratio": orig_shape_metrics.get("aspect_ratio", 0) if orig_shape_metrics else 0,

            

            "sit3_enh_circularity": enh_shape_metrics.get("circularity", 0) if enh_shape_metrics else 0,

            "sit3_enh_solidity": enh_shape_metrics.get("solidity", 0) if enh_shape_metrics else 0,

            "sit3_enh_aspect_ratio": enh_shape_metrics.get("aspect_ratio", 0) if enh_shape_metrics else 0

        },

        "ripeness": {

            "orig_fg_ratio": fg_ratio_orig_roi,

            "enh_fg_ratio": fg_ratio_enh_roi,

            

            "orig_hsv": {"h": float(s1_orig_mean_h), "s": float(s1_orig_mean_s), "v": float(s1_orig_mean_v)},

            "enh_hsv": {"h": float(s1_enh_mean_h), "s": float(s1_enh_mean_s), "v": float(s1_enh_mean_v)},

            

            "orig_lab": {"l": float(s1_orig_mean_l), "a": float(s1_orig_mean_a), "b": float(s1_orig_mean_b)},

            "enh_lab": {"l": float(s1_enh_mean_l), "a": float(s1_enh_mean_a), "b": float(s1_enh_mean_b)}

        }

    }



    return results



import tempfile

import os

import math



@app.route('/api/inspect', methods=['POST'])

def inspect_fruit():

    if 'image' not in request.files:

        return jsonify({"error": "No image uploaded"}), 400

        

    file = request.files['image']

    img_bytes = file.read()

    

                  

    nparr = np.frombuffer(img_bytes, np.uint8)

    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    

    if img is None:

        return jsonify({"error": "Invalid image file"}), 400

        

    results = process_image_frame(img)

    return jsonify(results)



@app.route('/api/inspect_video', methods=['POST'])

def inspect_video():

    if 'video' not in request.files:

        return jsonify({"error": "No video uploaded"}), 400

        

    file = request.files['video']

    

                               

    temp_dir = tempfile.gettempdir()

    temp_path = os.path.join(temp_dir, 'temp_video_upload.mp4')

    file.save(temp_path)

        

    cap = cv2.VideoCapture(temp_path)

    if not cap.isOpened():

        return jsonify({"error": "Could not open video file"}), 400

        

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0 or math.isnan(fps):

        fps = 30           

        

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = total_frames / fps if fps > 0 else 0

    

    print(f"\n--- Video Processing Started ---")

    print(f"Video Duration: {duration:.2f} seconds")

    print(f"Total Frames: {total_frames}")

    print(f"FPS: {fps:.2f}")

    

                                                    

    num_samples = min(5, total_frames)

    if num_samples == 0:

        return jsonify({"error": "Video contains no frames"}), 400

        

    if num_samples == 1:

        target_indices = [0]

    else:

                                                     

        target_indices = [int(i * (total_frames - 1) / (num_samples - 1)) for i in range(num_samples)]

        

    print(f"Selected Frame Positions: {target_indices}")

    

    frames_results = []

    

    total_start_time = time.time()

    

    for idx in target_indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

        ret, frame = cap.read()

        if not ret:

            print(f"Warning: Could not read frame at index {idx}")

            continue

            

        print(f"Processing frame at index {idx}...")

        frame_start_time = time.time()

        

        frame_res = process_image_frame(frame)

        if frame_res.get("success", False):

            frames_results.append(frame_res)

            

        frame_end_time = time.time()

        print(f"Frame {idx} processed in {frame_end_time - frame_start_time:.2f} seconds")

            

    cap.release()

    try:

        os.remove(temp_path)

    except:

        pass

        

    total_end_time = time.time()

    print(f"Total Video Processing Time: {total_end_time - total_start_time:.2f} seconds")

    print(f"--- Video Processing Finished ---\n")

        

    if not frames_results:

        return jsonify({"error": "No valid fruit frames found in video"}), 400

        

    return jsonify({"is_video": True, "frames": frames_results})





if __name__ == '__main__':

    app.run(host='127.0.0.1', port=5050, debug=True)

