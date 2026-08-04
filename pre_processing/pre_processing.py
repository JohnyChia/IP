import os
import cv2
from tqdm import tqdm

"""Preprocessing script using Ground Truth Labels
Reads images and labels from dataset/{train,valid,test}/.
If the label file is missing or empty (Noise Image), it is skipped.
If the label file has boxes, it extracts the largest bounding box (the main fruit),
crops the image tightly around it, and saves it to output/fruit.
This perfectly removes human faces and unnecessary backgrounds without AI.
"""
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_PATH, "dataset")
OUTPUT_PATH = os.path.join(BASE_PATH, "output", "fruit")
DATASETS = ["train", "valid", "test"]


def ensure_output():
    os.makedirs(OUTPUT_PATH, exist_ok=True)


def process_image(img_path, lbl_path):
    if not os.path.exists(lbl_path) or os.path.getsize(lbl_path) == 0:
        return
    with open(lbl_path, "r") as f:
        lines = f.readlines()
    if not lines:
        return
    img = cv2.imread(img_path)
    if img is None:
        return
    h, w = img.shape[:2]
    largest_box = None
    max_area = 0
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            _, x_c, y_c, bw, bh = map(float, parts[:5])
            area = bw * bh
            if area > max_area:
                max_area = area
                largest_box = (x_c, y_c, bw, bh)
    if largest_box is None:
        return
    x_c, y_c, bw, bh = largest_box
    box_w = int(bw * w)
    box_h = int(bh * h)
    box_x = int((x_c * w) - (box_w / 2))
    box_y = int((y_c * h) - (box_h / 2))
    box_x = max(0, box_x)
    box_y = max(0, box_y)
    box_w = min(w - box_x, box_w)
    box_h = min(h - box_y, box_h)
    if box_w <= 0 or box_h <= 0:
        return
    cropped_fruit = img[box_y : box_y + box_h, box_x : box_x + box_w]
    base = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(OUTPUT_PATH, base + ".jpg")
    cv2.imwrite(out_path, cropped_fruit)


def process_all():
    ensure_output()
    for ds in DATASETS:
        img_folder = os.path.join(DATASET_PATH, ds, "images")
        lbl_folder = os.path.join(DATASET_PATH, ds, "labels")
        if not os.path.isdir(img_folder) or not os.path.isdir(lbl_folder):
            continue
        images = []
        for f in os.listdir(img_folder):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                base = os.path.splitext(f)[0]
                img_path = os.path.join(img_folder, f)
                lbl_path = os.path.join(lbl_folder, base + ".txt")
                images.append((img_path, lbl_path))
        print(f"Processing {ds}: {len(images)} images (Cropping using Labels)...")
        for img_path, lbl_path in tqdm(images):
            process_image(img_path, lbl_path)


if __name__ == "__main__":
    process_all()
