import os
import pandas as pd

# هذا يعطي المسار المطلق لمجلد المشروع
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# هنا نحدد المسار النهائي للقالب
OUT_PATH = os.path.join(BASE_DIR, "dataset", "csv", "samples.csv")

columns = [
    "image_name",
    "label",               # Pure / Adulterated
    "density",             # g/ml
    "ph",                  # pH
    "flow_time",           # seconds
    "adulterant_type",     # None / Water / Sugar / Both (optional)
    "adulteration_percent" # 0..100 (optional)
]

# إنشاء المجلدات إذا لم تكن موجودة
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# إنشاء CSV فارغ بالقالب
df = pd.DataFrame(columns=columns)
df.to_csv(OUT_PATH, index=False, sep=";")

print(f"✅ Template created: {OUT_PATH}")
print("Columns:", columns)