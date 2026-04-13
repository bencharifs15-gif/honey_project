# data_tools/03_extract_features.py

import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

# ---------------- SETTINGS ----------------
SEP = ";"  # CSV separator

# قاعدة المشروع (يعتمد على موقع السكربت)
BASE_DIR = Path(__file__).resolve().parent.parent

# مسارات البيانات
IMG_ROOT   = BASE_DIR / "dataset" / "processed" / "segmented"   # الصور بعد augmentation + segmentation
SPLIT_DIR  = BASE_DIR / "dataset" / "csv" / "splits"            # CSVات train/test
OUT_DIR    = BASE_DIR / "dataset" / "features"                  # مكان حفظ الميزات
OUT_DIR.mkdir(parents=True, exist_ok=True)

# إعداد الجهاز
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_MAP = {"Pure": 0, "Adulterated": 1}

# ---------------- MODELS ----------------
def build_encoder():
    """MobileNetV2 pretrained, نستخدمه لاستخراج embeddings"""
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    m.classifier = nn.Identity()  # نأخذ embedding قبل التصنيف
    m.eval()
    return m.to(DEVICE)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# ---------------- HELPERS ----------------
@torch.no_grad()
def image_to_vec(model, img_path: Path) -> np.ndarray:
    """تحويل صورة إلى embedding"""
    img = Image.open(img_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(DEVICE)
    feat = model(x).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return feat

def load_split(csv_path: Path) -> pd.DataFrame:
    """قراءة CSV والتأكد من الأعمدة المطلوبة"""
    df = pd.read_csv(csv_path, sep=SEP)
    needed = ["image_name", "label", "density", "ph", "flow_time"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c} in {csv_path}")
    return df

def resolve_image_path(row) -> Path:
    """تحديد مسار الصورة داخل مجلد segmented"""
    label = row["label"]
    sub = "pure" if label == "Pure" else "adulterated"
    return IMG_ROOT / sub / str(row["image_name"])

# ---------------- FEATURE EXTRACTION ----------------
def extract_for_split(model, df: pd.DataFrame, out_npz: Path):
    """استخراج الميزات للصور مع البيانات الرقمية"""
    X_img = []
    X_num = []
    y = []
    names = []

    missing = 0
    for _, r in df.iterrows():
        img_path = resolve_image_path(r)
        if not img_path.exists():
            missing += 1
            continue

        # embeddings
        vec = image_to_vec(model, img_path)
        X_img.append(vec)

        # ميزات رقمية
        X_num.append([
            float(r["density"]),
            float(r["ph"]),
            float(r["flow_time"]),
        ])

        y.append(LABEL_MAP[str(r["label"])])
        names.append(str(r["image_name"]))

    if not X_img:
        raise RuntimeError(f"No images found for split! Check paths in CSV and {IMG_ROOT}")

    X_img = np.stack(X_img, axis=0)
    X_num = np.array(X_num, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    names = np.array(names)

    np.savez_compressed(out_npz, X_img=X_img, X_num=X_num, y=y, image_name=names)

    print(f"✅ Saved: {out_npz}")
    print("Shapes:", "X_img", X_img.shape, "| X_num", X_num.shape, "| y", y.shape)
    if missing:
        print(f"⚠️ Missing images skipped: {missing}")

# ---------------- MAIN ----------------
def main():
    train_csv = SPLIT_DIR / "train.csv"
    test_csv  = SPLIT_DIR / "test.csv"

    # تحميل النموذج
    model = build_encoder()

    # قراءة CSV
    train_df = load_split(train_csv)
    test_df  = load_split(test_csv)

    # استخراج الميزات
    extract_for_split(model, train_df, OUT_DIR / "train_features.npz")
    extract_for_split(model, test_df,  OUT_DIR / "test_features.npz")

    print("\nAll done ✅")

if __name__ == "__main__":
    main()