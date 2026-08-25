import os

import cv2

from tqdm import tqdm



DATASET_PATH = "dataset"

TRAIN_PATH = os.path.join(DATASET_PATH, "train")

VALID_PATH = os.path.join(DATASET_PATH, "valid")

TEST_PATH = os.path.join(DATASET_PATH, "test")

OUTPUT_FOLDER = "output"

OUTPUTS = [

    "resized",

    "denoise",

    "hsv",

    "mask",

    "morphology",

    "contour",

    "fruit_only",

    "brightness",

]





def create_output_folder():

    if not os.path.exists(OUTPUT_FOLDER):

        os.mkdir(OUTPUT_FOLDER)

    for folder in OUTPUTS:

        path = os.path.join(OUTPUT_FOLDER, folder)

        if not os.path.exists(path):

            os.mkdir(path)





def count_images(folder):

    total = 0

    for root, dirs, files in os.walk(folder):

        for file in files:

            if file.lower().endswith((".jpg", ".jpeg", ".png")):

                total += 1

    return total





def test_dataset(path):

    print("\nChecking:", path)

    if not os.path.exists(path):

        print("Folder not found")

        return

    images = []

    for root, dirs, files in os.walk(path):

        for file in files:

            if file.lower().endswith((".jpg", ".jpeg", ".png")):

                images.append(os.path.join(root, file))

    print("Images found:", len(images))

    if len(images) > 0:

        sample = images[0]

        img = cv2.imread(sample)

        if img is not None:

            print("Sample image:", sample)

            print("Image size:", img.shape)

        else:

            print("Cannot read image")





if __name__ == "__main__":

    print("==============================")

    print(" Fruit Image Processing System ")

    print("==============================")

    create_output_folder()

    print("\nOutput folders created.")

    test_dataset(TRAIN_PATH)

    test_dataset(VALID_PATH)

    test_dataset(TEST_PATH)

    print("\nPart 1 completed successfully.")

