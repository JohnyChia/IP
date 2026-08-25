import os

import glob

import cv2

import pandas as pd

import numpy as np

from codes.ripeness_assessment import segment_roi, extract_hsv_features_roi, extract_lab_features_roi, classify_ripeness_hsv, classify_ripeness_lab

from src.visualization import generate_graphs



def load_yolo_gt(txt_path, img_w, img_h):

    if not os.path.exists(txt_path): return []

    boxes = []

    with open(txt_path, 'r') as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) >= 5:

                x_c, y_c, w, h = map(float, parts[1:5])

                x1 = int((x_c - w/2) * img_w)

                y1 = int((y_c - h/2) * img_h)

                x2 = int((x_c + w/2) * img_w)

                y2 = int((y_c + h/2) * img_h)

                

                x1 = max(0, x1)

                y1 = max(0, y1)

                x2 = min(img_w, x2)

                y2 = min(img_h, y2)

                

                if x2 > x1 and y2 > y1:

                    boxes.append([x1, y1, x2, y2])

    return boxes



def get_gt_path(img_name):

    base_name = os.path.splitext(img_name)[0] + ".txt"

    for split in ['train', 'test', 'valid']:

        path = os.path.join(r"C:\IP\dataset", split, "labels", base_name)

        if os.path.exists(path):

            return path

    return ""



def main():

    benchmark_dir = r"C:\IP\fruit_ripeness\data\benchmark"

    out_dir = r"C:\IP\fruit_ripeness\outputs"

    

    os.makedirs(out_dir, exist_ok=True)

    

    image_files = sorted(glob.glob(os.path.join(benchmark_dir, "*.jpg")))

    results_hsv = []

    results_lab = []

    

    for img_path in image_files:

        img_name = os.path.basename(img_path)

        img = cv2.imread(img_path)

        if img is None: continue

        

        h, w = img.shape[:2]

        txt_path = get_gt_path(img_name)

        gt_boxes = load_yolo_gt(txt_path, w, h)

        

        roi_count = len(gt_boxes)

        if roi_count == 0:

            continue

            

        valid_rois = 0

        total_fg_pixels = 0

        total_roi_pixels = 0

        

                               

        hsv_h_acc, hsv_s_acc, hsv_v_acc = 0.0, 0.0, 0.0

        lab_l_acc, lab_a_acc, lab_b_acc = 0.0, 0.0, 0.0

        

        time_hsv_total, time_lab_total = 0.0, 0.0

        

        for box in gt_boxes:

            x1, y1, x2, y2 = box

            box_w = x2 - x1

            box_h = y2 - y1

            

                              

            if box_w < 20 or box_h < 20:

                continue

                

            roi_img = img[y1:y2, x1:x2]

            total_roi_pixels += (box_w * box_h)

            

            fg_mask = segment_roi(roi_img)

            fg_count = np.count_nonzero(fg_mask)

            

            if fg_count == 0:

                continue

                

            valid_rois += 1

            total_fg_pixels += fg_count

            

                         

            mh, ms, mv, t_hsv = extract_hsv_features_roi(roi_img, fg_mask)

            hsv_h_acc += mh * fg_count

            hsv_s_acc += ms * fg_count

            hsv_v_acc += mv * fg_count

            time_hsv_total += t_hsv

            

                         

            ml, ma, mb, t_lab = extract_lab_features_roi(roi_img, fg_mask)

            lab_l_acc += ml * fg_count

            lab_a_acc += ma * fg_count

            lab_b_acc += mb * fg_count

            time_lab_total += t_lab

            

        segmentation_status = "Success" if valid_rois > 0 else "Failed"

        fg_ratio = round(total_fg_pixels / total_roi_pixels, 4) if total_roi_pixels > 0 else 0.0

        

        if valid_rois > 0:

            final_h = hsv_h_acc / total_fg_pixels

            final_s = hsv_s_acc / total_fg_pixels

            final_v = hsv_v_acc / total_fg_pixels

            

            final_L = lab_l_acc / total_fg_pixels

            final_a = lab_a_acc / total_fg_pixels

            final_b = lab_b_acc / total_fg_pixels

            

            class_hsv = classify_ripeness_hsv(final_h)

            class_lab = classify_ripeness_lab(final_a)

        else:

            final_h, final_s, final_v = None, None, None

            final_L, final_a, final_b = None, None, None

            class_hsv, class_lab = "Failed", "Failed"

            

                        

        results_hsv.append({

            'image_name': img_name,

            'roi_count': roi_count,

            'valid_roi_count': valid_rois,

            'segmentation_status': segmentation_status,

            'foreground_ratio': fg_ratio,

            'mean_hue': round(final_h, 4) if final_h is not None else None,

            'mean_saturation': round(final_s, 4) if final_s is not None else None,

            'mean_value': round(final_v, 4) if final_v is not None else None,

            'predicted_ripeness': class_hsv,

            'processing_time_s': round(time_hsv_total, 4)

        })

        

                        

        results_lab.append({

            'image_name': img_name,

            'roi_count': roi_count,

            'valid_roi_count': valid_rois,

            'segmentation_status': segmentation_status,

            'foreground_ratio': fg_ratio,

            'mean_L': round(final_L, 4) if final_L is not None else None,

            'mean_a': round(final_a, 4) if final_a is not None else None,

            'mean_b': round(final_b, 4) if final_b is not None else None,

            'predicted_ripeness': class_lab,

            'processing_time_s': round(time_lab_total, 4)

        })

        

    df_hsv = pd.DataFrame(results_hsv)

    df_lab = pd.DataFrame(results_lab)

    

    hsv_csv_path = os.path.join(out_dir, "ripeness_assessment_hsv.csv")

    lab_csv_path = os.path.join(out_dir, "ripeness_assessment_lab.csv")

    

    df_hsv.to_csv(hsv_csv_path, index=False)

    df_lab.to_csv(lab_csv_path, index=False)

    

    print(f"Aggregation complete. Processed {len(image_files)} images, resulting in 5 HSV rows and 5 LAB rows.")

    

                     

    generate_graphs(hsv_csv_path, lab_csv_path, os.path.join(out_dir, "graphs"))

    print("Graphs generated successfully.")



if __name__ == "__main__":

    main()

