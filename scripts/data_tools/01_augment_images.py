# data_tools/01_augment_images.py

import os
import cv2
import random
import numpy as np
from tqdm import tqdm

INPUT_DIR = "data"
OUTPUT_DIR = r"dataset\processed\augmented"  # raw string لتجنب escape

CLASSES = ["pure", "adulterated"]
IMG_SIZE = 256
AUG_PER_IMAGE = 10

def augment_image(image):
    results = []
    h, w = image.shape[:2]

    for _ in range(AUG_PER_IMAGE):
        img = image.copy()

        # 1️⃣ Rotation
        angle = random.uniform(-25, 25)
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        img = cv2.warpAffine(img, M, (w, h))

        # 2️⃣ Brightness (مع تجنب Overflow)
        value = random.randint(-40, 40)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + value, 0, 255)
        hsv = hsv.astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # 3️⃣ Contrast
        alpha = random.uniform(0.8, 1.2)
        img = cv2.convertScaleAbs(img, alpha=alpha)

        # 4️⃣ Horizontal flip
        if random.random() > 0.5:
            img = cv2.flip(img, 1)

        # 5️⃣ Simple crop
        max_crop_x = min(10, w - 1)
        max_crop_y = min(10, h - 1)
        x = random.randint(0, max_crop_x)
        y = random.randint(0, max_crop_y)
        img = img[y:h-max_crop_y, x:w-max_crop_x]

        # 6️⃣ Resize back to IMG_SIZE
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        results.append(img)

    return results

def main():
    print("🔄 Augmentation started...")

    for cls in CLASSES:
        input_path = os.path.join(INPUT_DIR, cls)
        output_path = os.path.join(OUTPUT_DIR, cls)

        # تأكد من إنشاء المجلد الخاص بكل class
        os.makedirs(output_path, exist_ok=True)

        for img_name in tqdm(os.listdir(input_path)):
            img_path = os.path.join(input_path, img_name)
            image = cv2.imread(img_path)
            if image is None:
                print(f"⚠️  Cannot read image: {img_path}")
                continue
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

            augmented_images = augment_image(image)

            for i, img in enumerate(augmented_images):
                save_name = f"{img_name.split('.')[0]}_aug_{i}.jpg"
                cv2.imwrite(os.path.join(output_path, save_name), img)

    print("✅ Augmentation done")

if __name__ == "__main__":
    main()