import os

import cv2

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from tqdm import tqdm

from skimage.metrics import structural_similarity as ssim



BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ORIGINAL_PATH = os.path.join(BASE_PATH, "dataset")

PROCESSED_PATH = os.path.join(BASE_PATH, "output", "fruit")

VISUALIZE_PATH = os.path.join(BASE_PATH, "output", "visualize")

GRAPH_PATH = os.path.join(VISUALIZE_PATH, "preprocessing_graph.png")

CSV_PATH = os.path.join(VISUALIZE_PATH, "preprocessing_result.csv")







def get_image_map(folder):

    valid_exts = (".jpg", ".jpeg", ".png")

    if not os.path.isdir(folder):

        return {}

    img_map = {}

    for root, _, files in os.walk(folder):

        for f in files:

            if f.lower().endswith(valid_exts):

                img_map[os.path.splitext(f)[0]] = os.path.join(root, f)

    return img_map





def composite_to_gray(img):

    if len(img.shape) == 3 and img.shape[2] == 4:

        bgr = img[:, :, :3].astype(float)

        alpha = img[:, :, 3].astype(float) / 255.0

        bg = np.full_like(bgr, 128, dtype=float)

        comp = (bgr * alpha[..., None]) + (bg * (1 - alpha[..., None]))

        return cv2.cvtColor(comp.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)





def get_alpha(img):

    if len(img.shape) == 3 and img.shape[2] == 4:

        return img[:, :, 3]

    return None





def calc_bg_removed(img):

    alpha = get_alpha(img)

    if alpha is not None:

        return (alpha == 0).sum() / alpha.size * 100

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bg_pixels = gray.size - cv2.countNonZero(gray)

    return bg_pixels / gray.size * 100





def calc_laplacian_var(gray_img, mask=None):

    lap = cv2.Laplacian(gray_img, cv2.CV_64F)

    if mask is not None:

        vals = lap[mask]

        return float(np.var(vals)) if vals.size > 0 else 0.0

    return float(np.var(lap))





def calc_noise_reduction(orig_gray, proc_gray):

    before = calc_laplacian_var(orig_gray)

    after = calc_laplacian_var(proc_gray)

    return ((before - after) / before * 100) if before > 0 else 0.0





def evaluate_images():

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    orig_map = get_image_map(ORIGINAL_PATH)

    proc_map = get_image_map(PROCESSED_PATH)

    common_stems = sorted(list(set(orig_map.keys()) & set(proc_map.keys())))

    if not common_stems:

        print("No matching images found to compare.")

        return None

    results = []

    for stem in tqdm(common_stems, desc="Evaluating images"):

        orig = cv2.imread(orig_map[stem], cv2.IMREAD_COLOR)

        proc = cv2.imread(proc_map[stem], cv2.IMREAD_UNCHANGED)

        if orig is None or proc is None:

            continue

        orig_gray = composite_to_gray(orig)

        proc_gray = composite_to_gray(proc)

        orig_alpha = get_alpha(orig)

        proc_alpha = get_alpha(proc)

        orig_mask = (

            (orig_alpha > 10)

            if orig_alpha is not None

            else np.ones_like(orig_gray, dtype=bool)

        )

        proc_mask = (

            (proc_alpha > 10)

            if proc_alpha is not None

            else np.ones_like(proc_gray, dtype=bool)

        )

        try:

            ssim_proc_gray = proc_gray

            if ssim_proc_gray.shape != orig_gray.shape:

                ssim_proc_gray = cv2.resize(

                    ssim_proc_gray, (orig_gray.shape[1], orig_gray.shape[0])

                )

            ssim_val = float(ssim(orig_gray, ssim_proc_gray))

        except Exception:

            ssim_val = None

        results.append(

            {

                "filename": os.path.basename(orig_map[stem]),

                "background_removed_percent": calc_bg_removed(proc),

                "noise_reduction_percent": calc_noise_reduction(orig_gray, proc_gray),

                "original_laplacian": calc_laplacian_var(orig_gray),

                "processed_laplacian": calc_laplacian_var(proc_gray),

                "foreground_original_laplacian": calc_laplacian_var(

                    orig_gray, orig_mask

                ),

                "foreground_processed_laplacian": calc_laplacian_var(

                    proc_gray, proc_mask

                ),

                "ssim": ssim_val,

            }

        )

    df = pd.DataFrame(results)

    df.to_csv(CSV_PATH, index=False)

    print(f"Saved evaluation CSV: {CSV_PATH}")

    return df





def generate_graph(df):

    if df is None or df.empty:

        return

    plt.figure(figsize=(16, 12))

    ax1 = plt.subplot(2, 2, 1)

    ax1.axis("off")

    metrics = [

        ["Metric", "Value"],

        ["Images Tested", str(len(df))],

        ["Background Removed", f"{df['background_removed_percent'].mean():.2f}%"],

        ["Average SSIM", f"{df['ssim'].mean():.4f}"],

        ["Original FG Sharpness", f"{df['foreground_original_laplacian'].mean():.2f}"],

        [

            "Processed FG Sharpness",

            f"{df['foreground_processed_laplacian'].mean():.2f}",

        ],

    ]

    table = ax1.table(cellText=metrics, loc="center", cellLoc="center")

    table.scale(1, 2)

    ax1.set_title("Pre-processing Evaluation Summary")

    ax2 = plt.subplot(2, 2, 2)

    ax2.bar(

        ["Original FG", "Processed FG"],

        [

            df["foreground_original_laplacian"].mean(),

            df["foreground_processed_laplacian"].mean(),

        ],

    )

    ax2.set_title("Foreground Sharpness Comparison")

    ax2.set_ylabel("Laplacian Variance")

    ax3 = plt.subplot(2, 2, 3)

    sample = df.head(100)

    ax3.plot(

        sample.index, sample["background_removed_percent"], label="Background Removal"

    )

    ax3.plot(

        sample.index,

        sample.get("ssim", pd.Series(np.zeros(len(sample)))) * 100,

        label="SSIM (x100)",

    )

    ax3.set_title("Pre-processing Performance Trend")

    ax3.set_xlabel("Image Index")

    ax3.set_ylabel("Percentage (%)")

    ax3.legend()

    ax4 = plt.subplot(2, 2, 4)

    cols = [c for c in df.columns if c != "filename"]

    sns.heatmap(

        df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax4

    )

    ax4.set_title("Feature Correlation Matrix")

    plt.tight_layout()

    plt.savefig(GRAPH_PATH, dpi=300, bbox_inches="tight")

    plt.close()

    print(f"Saved evaluation graph: {GRAPH_PATH}")





if __name__ == "__main__":

    df = evaluate_images()

    generate_graph(df)

