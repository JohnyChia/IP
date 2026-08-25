import os

import cv2

import numpy as np

import pandas as pd

import sys



from codes.situation1_segmentation import color_thresholding, lab_kmeans

from codes.situation2_boundary import canny_edge_detection, morphological_watershed

from codes.situation3_inspection import global_shape_descriptors, run_active_contour

from src.evaluation import evaluate_situation1, evaluate_situation2, evaluate_situation3_shape, calculate_iou

from src.visualization import save_comparison_figure, generate_quantitative_graphs



def load_yolo_gt(txt_path, img_w, img_h):

    if not os.path.exists(txt_path):

        return []

    boxes = []

    with open(txt_path, 'r') as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) >= 5:

                x_c, y_c, w, h = map(float, parts[1:5])

                x1 = (x_c - w/2) * img_w

                y1 = (y_c - h/2) * img_h

                x2 = (x_c + w/2) * img_w

                y2 = (y_c + h/2) * img_h

                boxes.append([x1, y1, x2, y2])

    return boxes



def calculate_map(all_pred_boxes, all_gt_boxes, iou_thresholds):

    aps = []

    for iou_thresh in iou_thresholds:

        tps, fps = [], []

        total_gt = sum(len(gts) for gts in all_gt_boxes)

        if total_gt == 0:

            aps.append(0.0)

            continue

        for preds, gts in zip(all_pred_boxes, all_gt_boxes):

            matched_gt = set()

            preds_sorted = sorted(preds, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)

            for pbox in preds_sorted:

                best_iou = 0

                best_gt_idx = -1

                for g_idx, gbox in enumerate(gts):

                    if g_idx in matched_gt: continue

                    iou = calculate_iou(pbox, gbox)

                    if iou > best_iou:

                        best_iou = iou

                        best_gt_idx = g_idx

                if best_iou >= iou_thresh:

                    tps.append(1)

                    fps.append(0)

                    matched_gt.add(best_gt_idx)

                else:

                    tps.append(0)

                    fps.append(1)

        if not tps:

            aps.append(0.0)

            continue

        tps = np.cumsum(tps)

        fps = np.cumsum(fps)

        recalls = tps / total_gt

        precisions = tps / (tps + fps)

        for i in range(len(precisions)-2, -1, -1):

            precisions[i] = max(precisions[i], precisions[i+1])

        ap = 0.0

        prev_r = 0.0

        for p, r in zip(precisions, recalls):

            ap += p * (r - prev_r)

            prev_r = r

        aps.append(ap)

    return aps[0], sum(aps)/len(aps)                 



def get_gt_path(img_name):

    base_name = os.path.splitext(img_name)[0] + ".txt"

    for split in ['train', 'test', 'valid']:

        path = os.path.join(r"C:\IP\dataset", split, "labels", base_name)

        if os.path.exists(path):

            return path

    return ""



def preflight_check(image_files, data_dir):

    print("==================================================")

    print("PREFLIGHT GT CHECK")

    print("==================================================")

    all_good = True

    for img_name in image_files:

        txt_path = get_gt_path(img_name)

        if not os.path.exists(txt_path):

            print(f"FAIL: Missing GT for {img_name}")

            all_good = False

        else:

            print(f"Found GT: {img_name} -> {txt_path}")

            

    if not all_good:

        print("HALTING EXECUTION due to missing Ground Truth files.")

        sys.exit(1)

    print("PREFLIGHT CHECK PASS\\n")



def process_situation1(image_files, data_dir, out_dir):

    results = []

    

    alg1_preds, alg2_preds, all_gts = [], [], []

    total_tp_ct, total_fp_ct, total_fn_ct = 0, 0, 0

    total_tp_km, total_fp_km, total_fn_km = 0, 0, 0

    total_pred_ct, total_pred_km = 0, 0

    total_gt = 0

    

    for img_name in image_files:

        img_path = os.path.join(data_dir, img_name)

        img = cv2.imread(img_path)

        if img is None: continue

        

        txt_path = get_gt_path(img_name)

        gt_boxes = load_yolo_gt(txt_path, img.shape[1], img.shape[0])

        all_gts.append(gt_boxes)

        total_gt += len(gt_boxes)

        

        mask_ct, _, time_ct = color_thresholding(img)

        eval_ct = evaluate_situation1(mask_ct, img.shape, img_name, gt_boxes)

        mask_km, _, time_km = lab_kmeans(img)

        eval_km = evaluate_situation1(mask_km, img.shape, img_name, gt_boxes)

        

        alg1_preds.append(eval_ct.pop('raw_pred_boxes'))

        alg2_preds.append(eval_km.pop('raw_pred_boxes'))

        

        total_pred_ct += eval_ct['prediction_count']

        total_tp_ct += eval_ct['tp']

        total_fp_ct += eval_ct['fp']

        total_fn_ct += eval_ct['fn']

        

        total_pred_km += eval_km['prediction_count']

        total_tp_km += eval_km['tp']

        total_fp_km += eval_km['fp']

        total_fn_km += eval_km['fn']

        

        eval_ct.pop('raw_gt_boxes')

        eval_km.pop('raw_gt_boxes')

        

        os.makedirs(os.path.join(out_dir, "graphs", "situation1"), exist_ok=True)

        save_comparison_figure(os.path.join(out_dir, "graphs", "situation1", f"S1_{os.path.splitext(img_name)[0]}_comparison.jpg"), [(img, "Original"), (mask_ct, "Color Thresholding"), (mask_km, "LAB K-Means")])

        

        results.extend([{'image_name': img_name, 'algorithm': 'Color Thresholding', 'processing_time_s': round(time_ct, 4), **{k: round(v, 4) if isinstance(v, float) else v for k, v in eval_ct.items()}},

                        {'image_name': img_name, 'algorithm': 'LAB K-Means', 'processing_time_s': round(time_km, 4), **{k: round(v, 4) if isinstance(v, float) else v for k, v in eval_km.items()}}])

        

    df = pd.DataFrame(results)[['image_name', 'algorithm', 'processing_time_s', 'gt_count', 'prediction_count', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'mean_matched_iou', 'mean_prediction_overlap', 'mask_area_px', 'foreground_ratio', 'contour_count', 'largest_contour_area']]

    

    iou_thresholds = np.arange(0.5, 1.00, 0.05)

    ct_map50, ct_map50_95 = calculate_map(alg1_preds, all_gts, iou_thresholds)

    km_map50, km_map50_95 = calculate_map(alg2_preds, all_gts, iou_thresholds)

    

    ct_prec = total_tp_ct / (total_tp_ct + total_fp_ct) if (total_tp_ct + total_fp_ct) > 0 else 0.0

    ct_rec = total_tp_ct / (total_tp_ct + total_fn_ct) if (total_tp_ct + total_fn_ct) > 0 else 0.0

    ct_f1 = 2 * ct_prec * ct_rec / (ct_prec + ct_rec) if (ct_prec + ct_rec) > 0 else 0.0

    ct_miou = df[df['algorithm'] == 'Color Thresholding']['mean_matched_iou'].mean()

    

    km_prec = total_tp_km / (total_tp_km + total_fp_km) if (total_tp_km + total_fp_km) > 0 else 0.0

    km_rec = total_tp_km / (total_tp_km + total_fn_km) if (total_tp_km + total_fn_km) > 0 else 0.0

    km_f1 = 2 * km_prec * km_rec / (km_prec + km_rec) if (km_prec + km_rec) > 0 else 0.0

    km_miou = df[df['algorithm'] == 'LAB K-Means']['mean_matched_iou'].mean()



    summary_df = pd.DataFrame([

        {'situation': 'Situation 1', 'algorithm': 'Color Thresholding', 'total_images': 5, 'total_gt': total_gt, 'total_predictions': total_pred_ct, 'total_tp': total_tp_ct, 'total_fp': total_fp_ct, 'total_fn': total_fn_ct, 'precision': round(ct_prec, 4), 'recall': round(ct_rec, 4), 'f1': round(ct_f1, 4), 'mean_matched_iou': round(ct_miou, 4), 'map50': round(ct_map50, 4), 'map50_95': round(ct_map50_95, 4), 'avg_processing_time_s': round(df[df['algorithm'] == 'Color Thresholding']['processing_time_s'].mean(), 4)},

        {'situation': 'Situation 1', 'algorithm': 'LAB K-Means', 'total_images': 5, 'total_gt': total_gt, 'total_predictions': total_pred_km, 'total_tp': total_tp_km, 'total_fp': total_fp_km, 'total_fn': total_fn_km, 'precision': round(km_prec, 4), 'recall': round(km_rec, 4), 'f1': round(km_f1, 4), 'mean_matched_iou': round(km_miou, 4), 'map50': round(km_map50, 4), 'map50_95': round(km_map50_95, 4), 'avg_processing_time_s': round(df[df['algorithm'] == 'LAB K-Means']['processing_time_s'].mean(), 4)}

    ])

    

    df.to_csv(os.path.join(out_dir, "situation1.csv"), index=False)

    return df, summary_df



def process_situation2(image_files, data_dir, out_dir):

    results = []

    

    alg1_preds, alg2_preds, all_gts = [], [], []

    total_tp_c, total_fp_c, total_fn_c = 0, 0, 0

    total_tp_w, total_fp_w, total_fn_w = 0, 0, 0

    total_pred_c, total_pred_w = 0, 0

    total_gt = 0

    

    for img_name in image_files:

        img_path = os.path.join(data_dir, img_name)

        img = cv2.imread(img_path)

        if img is None: continue

        

        txt_path = get_gt_path(img_name)

        gt_boxes = load_yolo_gt(txt_path, img.shape[1], img.shape[0])

        all_gts.append(gt_boxes)

        total_gt += len(gt_boxes)

        

        mask_c, _, time_c = canny_edge_detection(img)

        eval_c = evaluate_situation2(mask_c, img.shape, img_name, gt_boxes)

        mask_w, _, time_w = morphological_watershed(img)

        eval_w = evaluate_situation2(mask_w, img.shape, img_name, gt_boxes)

        

        alg1_preds.append(eval_c.pop('raw_pred_boxes'))

        alg2_preds.append(eval_w.pop('raw_pred_boxes'))

        

        total_pred_c += eval_c['prediction_count']

        total_tp_c += eval_c['tp']

        total_fp_c += eval_c['fp']

        total_fn_c += eval_c['fn']

        

        total_pred_w += eval_w['prediction_count']

        total_tp_w += eval_w['tp']

        total_fp_w += eval_w['fp']

        total_fn_w += eval_w['fn']

        

        eval_c.pop('raw_gt_boxes')

        eval_w.pop('raw_gt_boxes')

        

        os.makedirs(os.path.join(out_dir, "graphs", "situation2"), exist_ok=True)

        save_comparison_figure(os.path.join(out_dir, "graphs", "situation2", f"S2_{os.path.splitext(img_name)[0]}_comparison.jpg"), [(img, "Original"), (mask_c, "Canny Edge"), (mask_w, "Morphological Watershed")])

        

        results.extend([{'image_name': img_name, 'algorithm': 'Canny Edge Detection', 'processing_time_s': round(time_c, 4), **{k: round(v, 4) if isinstance(v, float) else v for k, v in eval_c.items()}},

                        {'image_name': img_name, 'algorithm': 'Morphological Watershed', 'processing_time_s': round(time_w, 4), **{k: round(v, 4) if isinstance(v, float) else v for k, v in eval_w.items()}}])

        

    df = pd.DataFrame(results)[['image_name', 'algorithm', 'processing_time_s', 'gt_count', 'prediction_count', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'mean_matched_iou', 'mean_prediction_overlap', 'edge_area_px', 'edge_density', 'boundary_area', 'contour_count']]

    

    iou_thresholds = np.arange(0.5, 1.00, 0.05)

    c_map50, c_map50_95 = calculate_map(alg1_preds, all_gts, iou_thresholds)

    w_map50, w_map50_95 = calculate_map(alg2_preds, all_gts, iou_thresholds)

    

    c_prec = total_tp_c / (total_tp_c + total_fp_c) if (total_tp_c + total_fp_c) > 0 else 0.0

    c_rec = total_tp_c / (total_tp_c + total_fn_c) if (total_tp_c + total_fn_c) > 0 else 0.0

    c_f1 = 2 * c_prec * c_rec / (c_prec + c_rec) if (c_prec + c_rec) > 0 else 0.0

    c_miou = df[df['algorithm'] == 'Canny Edge Detection']['mean_matched_iou'].mean()

    

    w_prec = total_tp_w / (total_tp_w + total_fp_w) if (total_tp_w + total_fp_w) > 0 else 0.0

    w_rec = total_tp_w / (total_tp_w + total_fn_w) if (total_tp_w + total_fn_w) > 0 else 0.0

    w_f1 = 2 * w_prec * w_rec / (w_prec + w_rec) if (w_prec + w_rec) > 0 else 0.0

    w_miou = df[df['algorithm'] == 'Morphological Watershed']['mean_matched_iou'].mean()



    summary_df = pd.DataFrame([

        {'situation': 'Situation 2', 'algorithm': 'Canny Edge Detection', 'total_images': 5, 'total_gt': total_gt, 'total_predictions': total_pred_c, 'total_tp': total_tp_c, 'total_fp': total_fp_c, 'total_fn': total_fn_c, 'precision': round(c_prec, 4), 'recall': round(c_rec, 4), 'f1': round(c_f1, 4), 'mean_matched_iou': round(c_miou, 4), 'map50': round(c_map50, 4), 'map50_95': round(c_map50_95, 4), 'avg_processing_time_s': round(df[df['algorithm'] == 'Canny Edge Detection']['processing_time_s'].mean(), 4)},

        {'situation': 'Situation 2', 'algorithm': 'Morphological Watershed', 'total_images': 5, 'total_gt': total_gt, 'total_predictions': total_pred_w, 'total_tp': total_tp_w, 'total_fp': total_fp_w, 'total_fn': total_fn_w, 'precision': round(w_prec, 4), 'recall': round(w_rec, 4), 'f1': round(w_f1, 4), 'mean_matched_iou': round(w_miou, 4), 'map50': round(w_map50, 4), 'map50_95': round(w_map50_95, 4), 'avg_processing_time_s': round(df[df['algorithm'] == 'Morphological Watershed']['processing_time_s'].mean(), 4)}

    ])

    

    df.to_csv(os.path.join(out_dir, "situation2.csv"), index=False)

    return df, summary_df



def mask_from_contour(shape, contour):

    m = np.zeros(shape[:2], dtype=np.uint8)

    if contour is not None and len(contour) > 0:

        cv2.drawContours(m, [contour], 0, 255, -1)

    return m



def process_situation3(image_files, data_dir, out_dir):

    results = []

    

    for img_name in image_files:

        img_path = os.path.join(data_dir, img_name)

        img = cv2.imread(img_path)

        if img is None: continue

        

        _, init_contour, _ = color_thresholding(img)

        

        _, time_g = global_shape_descriptors(img, init_contour)

        eval_g = evaluate_situation3_shape(init_contour)

        

        final_contour, time_ac = run_active_contour(img, init_contour)

        eval_ac = evaluate_situation3_shape(final_contour)

        

        mask_g = mask_from_contour(img.shape, init_contour)

        mask_ac = mask_from_contour(img.shape, final_contour)

        

        os.makedirs(os.path.join(out_dir, "graphs", "situation3"), exist_ok=True)

        save_comparison_figure(os.path.join(out_dir, "graphs", "situation3", f"S3_{os.path.splitext(img_name)[0]}_comparison.jpg"), [(img, "Original"), (mask_g, "Global Shape Descriptors"), (mask_ac, "Active Contour")])

        

        results.extend([{'image_name': img_name, 'algorithm': 'Global Shape Descriptors', 'processing_time_s': round(time_g, 4), **{k: round(v, 4) if isinstance(v, float) else v for k, v in eval_g.items()}},

                        {'image_name': img_name, 'algorithm': 'Active Contour', 'processing_time_s': round(time_ac, 4), **{k: round(v, 4) if isinstance(v, float) else v for k, v in eval_ac.items()}}])

        

    df = pd.DataFrame(results)[['image_name', 'algorithm', 'processing_time_s', 'area', 'perimeter', 'circularity', 'hull_area', 'solidity', 'aspect_ratio']]

    

    df.to_csv(os.path.join(out_dir, "situation3.csv"), index=False)

    return df



def validate_outputs(out_dir, image_files):

    print("\\n==================================================")

    print("              FINAL VALIDATION")

    print("==================================================\\n")

    

    def pf(cond, msg):

        status = 'PASS' if cond else 'FAIL'

        print(f"{msg}: {status}")

        return cond

        

    s1_df = pd.read_csv(os.path.join(out_dir, 'situation1.csv'))

    s2_df = pd.read_csv(os.path.join(out_dir, 'situation2.csv'))

    s3_df = pd.read_csv(os.path.join(out_dir, 'situation3.csv'))

    

    pf(len(s1_df) == 10 and len(s2_df) == 10 and len(s3_df) == 10, "situation1/2/3.csv each have exactly 10 rows")

    

                                 

    has_na = False

    for df in [s1_df, s2_df, s3_df]:

        if df.isnull().values.any(): has_na = True

    pf(not has_na, "No NA values in calculable metrics")

    

    pf(len(s1_df['image_name'].unique()) == 5, "Exactly 5 unique images in results")

    

    tp_fn_correct = True

    for df in [s1_df, s2_df]:

        if not (df['tp'] + df['fn'] == df['gt_count']).all():

            tp_fn_correct = False

    pf(tp_fn_correct, "TP + FN == GT count per image")

    

    pf(len(s1_df.columns) == 17, "Situation 1 columns match exact specification")

    pf(len(s2_df.columns) == 17, "Situation 2 columns match exact specification")

    pf(len(s3_df.columns) == 9, "Situation 3 columns match exact specification")



def print_system_module():

    print("==================================================")

    print("          FRUIT INSPECTION SYSTEM")

    print("==================================================\\n")

    print("[MODULE 1] FRUIT SURFACE DETECTION MODULE\\n")

    

def main():

    print_system_module()

    data_dir = r"C:\IP\fruit_surface_detection\data\benchmark"

    out_dir = r"C:\IP\fruit_surface_detection\outputs"

    os.makedirs(out_dir, exist_ok=True)

    

    image_files = sorted([f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.png'))])

    preflight_check(image_files, data_dir)

    

    print("Running Situation 1...")

    df1, s1_sum = process_situation1(image_files, data_dir, out_dir)

    print("Running Situation 2...")

    df2, s2_sum = process_situation2(image_files, data_dir, out_dir)

    print("Running Situation 3...")

    df3 = process_situation3(image_files, data_dir, out_dir)

    

    generate_quantitative_graphs([df1, df2, df3], os.path.join(out_dir, 'graphs'))

    

    validate_outputs(out_dir, image_files)

    

    print("\\nFinal CSV files created:")

    print("1. situation1.csv (10 rows)")

    print("2. situation2.csv (10 rows)")

    print("3. situation3.csv (10 rows)")

if __name__ == "__main__":

    main()

