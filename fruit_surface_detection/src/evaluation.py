import cv2

import numpy as np



def calculate_iou(box1, box2):

    x1, y1, x2, y2 = box1

    x3, y3, x4, y4 = box2

    xi1, yi1 = max(x1, x3), max(y1, y3)

    xi2, yi2 = min(x2, x4), min(y2, y4)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    if inter_area == 0: return 0.0

    box1_area = (x2 - x1) * (y2 - y1)

    box2_area = (x4 - x3) * (y4 - y3)

    return inter_area / float(box1_area + box2_area - inter_area)



def get_base_metrics(mask, shape, gt_boxes):

    metrics = {}

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    

    pred_boxes = []

    for c in contours:

        area = cv2.contourArea(c)

        if area > 10:

            x, y, bw, bh = cv2.boundingRect(c)

            pred_boxes.append([x, y, x + bw, y + bh])

            

    tp = 0

    fp = 0

    fn = 0

    matched_ious = []

    

    if len(gt_boxes) == 0:

        fp = len(pred_boxes)

    else:

        matched_gt = set()

        preds_sorted = sorted(pred_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)

        for pbox in preds_sorted:

            best_iou = 0

            best_gt_idx = -1

            for g_idx, gbox in enumerate(gt_boxes):

                if g_idx in matched_gt: continue

                iou = calculate_iou(pbox, gbox)

                if iou > best_iou:

                    best_iou = iou

                    best_gt_idx = g_idx

            if best_iou >= 0.50:

                tp += 1

                matched_gt.add(best_gt_idx)

                matched_ious.append(best_iou)

            else:

                fp += 1

        fn = len(gt_boxes) - len(matched_gt)

        

    metrics['gt_count'] = len(gt_boxes)

    metrics['prediction_count'] = len(pred_boxes)

    metrics['tp'] = tp

    metrics['fp'] = fp

    metrics['fn'] = fn

    

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    mean_matched_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0

    

    best_ious_per_pred = []

    if len(gt_boxes) > 0 and len(pred_boxes) > 0:

        for pbox in pred_boxes:

            max_iou = 0.0

            for gbox in gt_boxes:

                iou = calculate_iou(pbox, gbox)

                if iou > max_iou:

                    max_iou = iou

            best_ious_per_pred.append(max_iou)

    mean_prediction_overlap = sum(best_ious_per_pred) / len(best_ious_per_pred) if best_ious_per_pred else 0.0

    

    metrics['precision'] = float(precision)

    metrics['recall'] = float(recall)

    metrics['f1'] = float(f1)

    metrics['mean_matched_iou'] = float(mean_matched_iou)

    metrics['mean_prediction_overlap'] = float(mean_prediction_overlap)

    

    metrics['raw_pred_boxes'] = pred_boxes

    metrics['raw_gt_boxes'] = gt_boxes

    return metrics, contours



def evaluate_situation1(mask, shape, image_name, gt_boxes):

    metrics, contours = get_base_metrics(mask, shape, gt_boxes)

    mask_area = int(np.count_nonzero(mask))

    foreground_ratio = mask_area / (shape[0] * shape[1])

    

    largest_contour_area = max(cv2.contourArea(c) for c in contours) if contours else 0

    

    metrics['mask_area_px'] = mask_area

    metrics['foreground_ratio'] = foreground_ratio

    metrics['contour_count'] = len(contours)

    metrics['largest_contour_area'] = float(largest_contour_area)

    return metrics



def evaluate_situation2(mask, shape, image_name, gt_boxes):

    metrics, contours = get_base_metrics(mask, shape, gt_boxes)

    kernel = np.ones((3,3), np.uint8)

    mask_eroded = cv2.erode(mask, kernel, iterations=1)

    edge_area = int(np.count_nonzero(mask) - np.count_nonzero(mask_eroded))

    edge_density = float(edge_area / (shape[0] * shape[1]))

    

    boundary_area = sum(cv2.contourArea(c) for c in contours)

    

    metrics['edge_area_px'] = edge_area

    metrics['edge_density'] = edge_density

    metrics['boundary_area'] = float(boundary_area)

    metrics['contour_count'] = len(contours)

    return metrics



def evaluate_situation3_shape(contour):

    metrics = {

        'area': 0.0,

        'perimeter': 0.0,

        'circularity': 0.0,

        'hull_area': 0.0,

        'solidity': 0.0,

        'aspect_ratio': 0.0

    }

    if contour is not None and len(contour) >= 3:

        area = cv2.contourArea(contour)

        perimeter = cv2.arcLength(contour, True)

        

        circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0.0

        

        hull = cv2.convexHull(contour)

        hull_area = cv2.contourArea(hull)

        

        solidity = float(area) / hull_area if hull_area > 0 else 0.0

        

        x, y, w, h = cv2.boundingRect(contour)

        aspect_ratio = float(w) / h if h > 0 else 0.0

        

        metrics['area'] = float(area)

        metrics['perimeter'] = float(perimeter)

        metrics['circularity'] = float(circularity)

        metrics['hull_area'] = float(hull_area)

        metrics['solidity'] = float(solidity)

        metrics['aspect_ratio'] = float(aspect_ratio)

        

    return metrics

