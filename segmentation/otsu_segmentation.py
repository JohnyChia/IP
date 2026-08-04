import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_PATH, "output", "fruit")
OUTPUT_PATH = os.path.join(BASE_PATH, "output", "segmented_defects")
GRAPH_PATH = os.path.join(BASE_PATH, "evaluation", "segmentation_graph.png")


def ensure_output():
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, "masks"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, "overlays"), exist_ok=True)
    os.makedirs(os.path.dirname(GRAPH_PATH), exist_ok=True)


def process_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    s_channel = hsv[:, :, 1]
    a_channel = lab[:, :, 1]
    intensity_channel = cv2.addWeighted(s_channel, 0.5, a_channel, 0.5, 0)
    optimal_T, binary_mask = cv2.threshold(
        intensity_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(
        binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    for i in range(len(contours)):
        cv2.drawContours(binary_mask, contours, i, 255, -1)
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cleaned_mask = np.zeros_like(binary_mask)
        cv2.drawContours(cleaned_mask, [largest_contour], 0, 255, -1)
        binary_mask = cleaned_mask
    overlay = img.copy()
    red_overlay = np.zeros_like(overlay)
    red_overlay[:] = [0, 0, 255]
    fruit_pixels = binary_mask == 255
    overlay[fruit_pixels] = cv2.addWeighted(
        overlay[fruit_pixels], 0.5, red_overlay[fruit_pixels], 0.5, 0
    )
    base = os.path.splitext(os.path.basename(img_path))[0]
    cv2.imwrite(os.path.join(OUTPUT_PATH, "masks", base + "_mask.png"), binary_mask)
    cv2.imwrite(os.path.join(OUTPUT_PATH, "overlays", base + "_overlay.jpg"), overlay)
    coverage = (fruit_pixels.sum() / binary_mask.size) * 100
    return coverage


def generate_graph(coverages):
    if not coverages:
        return
    plt.figure(figsize=(12, 5))
    under = sum(1 for c in coverages if c < 10)
    over = sum(1 for c in coverages if c > 95)
    healthy = sum(1 for c in coverages if 10 <= c <= 95)
    labels = ["Healthy (10%-95%)", "Over-segmented (>95%)", "Under-segmented (<10%)"]
    sizes = [healthy, over, under]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    sizes, labels, colors = zip(
        *[(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
    )
    ax1 = plt.subplot(1, 2, 1)
    sns.histplot(coverages, bins=50, kde=True, color="skyblue", ax=ax1)
    ax1.set_title("Fruit Coverage Percentage", fontsize=14)
    ax1.set_xlabel("Coverage Percentage (%)")
    ax1.set_ylabel("Number of Images")
    ax1.axvline(x=78.5, color="r", linestyle="--", label="Theoretical Ideal (78.5%)")
    ax1.legend()
    ax2 = plt.subplot(1, 2, 2)
    ax2.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
    ax2.set_title("Segmentation Quality", fontsize=14)
    plt.tight_layout()
    plt.savefig(GRAPH_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved evaluation graph: {GRAPH_PATH}")


def process_all():
    ensure_output()
    if not os.path.isdir(INPUT_PATH):
        print(f"Input directory not found: {INPUT_PATH}")
        return
    images = []
    for f in os.listdir(INPUT_PATH):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            images.append(os.path.join(INPUT_PATH, f))
    print(f"Running Full-Fruit Otsu Thresholding: {len(images)} images...")
    coverages = []
    for p in tqdm(images):
        cov = process_image(p)
        if cov is not None:
            coverages.append(cov)
    if coverages:
        print("Generating evaluation graph...")
        generate_graph(coverages)


if __name__ == "__main__":
    process_all()
