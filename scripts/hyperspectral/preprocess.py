import pandas as pd

df = pd.read_csv("data/hyperspectral/raw/adulteration_dataset.csv")

# تنظيف بسيط
df = df.dropna()

# حفظ نسخة نظيفة
df.to_csv("data/hyperspectral/processed/clean.csv", index=False)

print("✅ Preprocessing done")