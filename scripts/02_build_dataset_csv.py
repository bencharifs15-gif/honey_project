import os
import re
import random
from pathlib import Path
import pandas as pd

# -------- SETTINGS --------
IMG_ROOT = Path("dataset\processed\segmented")
OUT_CSV = Path("dataset/csv/samples.csv")
SEP = ";"          # you requested semicolon separator
random.seed(42)

def group_id_from_name(filename: str) -> str:
    """
    pure01_aug03.jpeg -> pure01
    adulterated05_aug09.jpeg -> adulterated05
    pure07.jpeg -> pure07
    """
    stem = Path(filename).stem
    return re.sub(r"_aug\d+$", "", stem, flags=re.IGNORECASE)

def sample_values(label: str):
    # plausible ranges (you can adjust later)
    if label == "Pure":
        density = random.uniform(1.38, 1.45)
        ph = random.uniform(3.4, 4.5)
        flow = random.uniform(10, 20)
        adulterant_type = ""
    else:
        density = random.uniform(1.25, 1.36)
        ph = random.uniform(4.6, 6.0)
        flow = random.uniform(4, 10)
        adulterant_type = "Sugar"

    # round to nice numbers
    return round(density, 2), round(ph, 2), int(round(flow)), adulterant_type

def main():
    rows = []
    base_cache = {}  # group_id -> (density, ph, flow_time, adulterant_type)

    for sub, label in [("pure", "Pure"), ("adulterated", "Adulterated")]:
        folder = IMG_ROOT / sub
        if not folder.exists():
            print(f"❌ Missing folder: {folder}")
            return

        files = [p for p in folder.iterdir() if p.suffix.lower() in [".jpg",".jpeg",".png"]]
        files.sort()

        print(f"📁 {sub}: {len(files)} images found")

        for p in files:
            image_name = p.name
            gid = group_id_from_name(image_name)

            if gid not in base_cache:
                base_cache[gid] = sample_values(label)

            density, ph, flow_time, adulterant_type = base_cache[gid]

            rows.append({
                "image_name": image_name,
                "label": label,
                "density": density,
                "ph": ph,
                "flow_time": flow_time,
                "adulterant_type": adulterant_type,
            })

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Save with semicolon separator
    df.to_csv(OUT_CSV, index=False, sep=SEP, encoding="utf-8")

    print(f"\n✅ CSV created: {OUT_CSV}")
    print("Columns:", list(df.columns))
    print("Rows:", len(df))

if __name__ == "__main__":
    main()
