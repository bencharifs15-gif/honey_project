# data_tools/02_segment_images.py

import os
import cv2
from tqdm import tqdm

INPUT_DIR = r"dataset/processed/augmented"  # مسار الصور المعززة
OUTPUT_DIR = r"dataset/processed/segmented"  # مجلد حفظ الباتشات

CLASSES = ["pure", "adulterated"]  # نفس الأسماء المستخدمة في augmentation
GRID_SIZE = 5  # تقسيم الصورة إلى 5x5

def segment_image(image):
    patches = []
    h, w = image.shape[:2]

    ph = h // GRID_SIZE
    pw = w // GRID_SIZE

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            patch = image[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
            patches.append(patch)

    return patches

def main():
    print("🔄 Segmentation started...")

    for cls in CLASSES:
        input_path = os.path.join(INPUT_DIR, cls)
        output_path = os.path.join(OUTPUT_DIR, cls)

        os.makedirs(output_path, exist_ok=True)

        for img_name in tqdm(os.listdir(input_path)):
            img_path = os.path.join(input_path, img_name)
            image = cv2.imread(img_path)
            if image is None:
                print(f"⚠️  Cannot read image: {img_path}")
                continue

            patches = segment_image(image)

            for i, patch in enumerate(patches):
                save_name = f"{img_name.split('.')[0]}_patch_{i}.jpg"
                cv2.imwrite(os.path.join(output_path, save_name), patch)

    print("✅ Segmentation done")

if __name__ == "__main__":
    main()