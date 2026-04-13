# data_tools/03_build_dataset_csv.py

import os
import pandas as pd

INPUT_DIR = r"dataset/processed/segmented"  # مسار الباتشات المصححة
CSV_PATH = "dataset.csv"

CLASSES = ["pure", "adulterated"]  # تطابق ما استخدمناه في segmentation

LABEL_MAP = {
    "pure": 0,
    "adulterated": 1
}

def main():
    print("🔄 Building dataset CSV...")

    data = []

    for cls in CLASSES:
        class_path = os.path.join(INPUT_DIR, cls)
        if not os.path.exists(class_path):
            print(f"⚠️ Class folder not found: {class_path}")
            continue

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            data.append({
                "image_path": img_path,
                "label": LABEL_MAP[cls],
                "class": cls
            })

    if not data:
        print("⚠️ No data found. Please check your segmented images folder.")
        return

    df = pd.DataFrame(data)

    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)

    df.to_csv(CSV_PATH, index=False)

    print("✅ CSV created successfully")
    print(df["class"].value_counts())
    print(f"Total samples: {len(df)}")


if __name__ == "__main__":
    main()