import pandas as pd
import os

df = pd.read_csv("data/hyperspectral/processed/clean.csv")

spectral_cols = [col for col in df.columns if 'nm' in col]

demo_dir = "data/hyperspectral/demo"
os.makedirs(demo_dir, exist_ok=True)

# نأخذ 5 عينات عشوائية
samples = df.sample(5)

for i, (_, row) in enumerate(samples.iterrows()):
    values = row[spectral_cols]

    values.to_csv(f"{demo_dir}/sample_{i}.csv", index=False, header=False)

print("✅ Demo CSV files created")