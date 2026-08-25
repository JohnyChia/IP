import os

import glob

import cv2

import numpy as np

import pandas as pd

import shutil



def parse_yolo_label(label_path, img_width, img_height):

    





       

    if not os.path.exists(label_path):

        return False, [], []

        

    is_segmentation = False

    polygons = []

    bboxes = []

    

    with open(label_path, 'r') as f:

        lines = f.readlines()

        for line in lines:

            parts = list(map(float, line.strip().split()))

            if len(parts) == 0:

                continue

            

                                  

            if len(parts) == 5:

                                                            

                _, x_c, y_c, w, h = parts

                x_c *= img_width

                y_c *= img_height

                w *= img_width

                h *= img_height

                bboxes.append([x_c, y_c, w, h])

            elif len(parts) > 5:

                                      

                is_segmentation = True

                poly_norm = parts[1:]

                poly_pixels = []

                for i in range(0, len(poly_norm), 2):

                    x = poly_norm[i] * img_width

                    y = poly_norm[i+1] * img_height

                    poly_pixels.append([x, y])

                polygons.append(poly_pixels)

                

    return is_segmentation, polygons, bboxes



def create_ground_truth_mask(img_shape, polygons, out_path):

    

       

    mask = np.zeros(img_shape[:2], dtype=np.uint8)

    for poly in polygons:

        pts = np.array(poly, np.int32)

        pts = pts.reshape((-1, 1, 2))

        cv2.fillPoly(mask, [pts], 255)

    cv2.imwrite(out_path, mask)



def categorize_images(images_dir, labels_dir):

    



       

    img_paths = glob.glob(os.path.join(images_dir, "*.jpg"))

    img_paths += glob.glob(os.path.join(images_dir, "*.png"))

    

    results = []

    

    for imp in img_paths:

        img_name = os.path.basename(imp)

        label_name = os.path.splitext(img_name)[0] + ".txt"

        label_path = os.path.join(labels_dir, label_name)

        

        img = cv2.imread(imp)

        if img is None:

            continue

            

        h, w = img.shape[:2]

        is_seg, polys, bboxes = parse_yolo_label(label_path, w, h)

        

                              

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mean_brightness = np.mean(gray)

        std_brightness = np.std(gray)

        

        edges = cv2.Canny(gray, 50, 150)

        edge_density = np.sum(edges > 0) / (h * w)

        

        results.append({

            'img_path': imp,

            'img_name': img_name,

            'is_segmentation': is_seg,

            'mean_brightness': mean_brightness,

            'std_brightness': std_brightness,

            'edge_density': edge_density,

            'polys': polys

        })

        

    df = pd.DataFrame(results)

    

                   

                                                                                      

    sit1 = df.sort_values(by='edge_density', ascending=True).head(5)

    

                                                                           

    sit2 = df.sort_values(by='std_brightness', ascending=False).head(5)

    

                                                                                         

    used = set(sit1['img_name']).union(set(sit2['img_name']))

    sit3_candidates = df[~df['img_name'].isin(used)]

    sit3 = sit3_candidates.sort_values(by='edge_density', ascending=False).head(5)

    

    return sit1, sit2, sit3



def setup_dataset():

    images_dir = r"C:\IP\dataset\train\images"

    labels_dir = r"C:\IP\dataset\train\labels"

    

    if not os.path.exists(images_dir):

        print(f"Warning: {images_dir} does not exist.")

        return

        

    sit1, sit2, sit3 = categorize_images(images_dir, labels_dir)

    

    situations = {

        'situation1': sit1,

        'situation2': sit2,

        'situation3': sit3

    }

    

    selection_records = []

    

    base_out = r"C:\IP\fruit_surface_detection\data"

    

    for sit_name, df_sit in situations.items():

        sit_dir = os.path.join(base_out, sit_name)

        os.makedirs(sit_dir, exist_ok=True)

        

        for _, row in df_sit.iterrows():

            img_path = row['img_path']

            img_name = row['img_name']

            

                        

            dst_img = os.path.join(sit_dir, img_name)

            shutil.copy(img_path, dst_img)

            

                                         

            gt_mask_path = ""

            if row['is_segmentation']:

                img = cv2.imread(img_path)

                mask_path = os.path.join(sit_dir, os.path.splitext(img_name)[0] + "_mask.png")

                create_ground_truth_mask(img.shape, row['polys'], mask_path)

                gt_mask_path = mask_path

                

            selection_records.append({

                'situation': sit_name,

                'image_name': img_name,

                'is_segmentation': row['is_segmentation'],

                'gt_mask_path': gt_mask_path

            })

            

    pd.DataFrame(selection_records).to_csv(os.path.join(base_out, "selected_images.csv"), index=False)

    print("Dataset setup complete. Images selected and masks generated where applicable.")



if __name__ == "__main__":

    setup_dataset()

