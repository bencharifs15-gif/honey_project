import re
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# -------- SETTINGS --------
IN_CSV = Path("dataset/csv/samples.csv")
OUT_DIR = Path("dataset/csv/splits")
SEP = ";"          # keep semicolon
TEST_SIZE = 0.2
RANDOM_STATE = 42

def group_id_from_name(filename: str) -> str:
    stem = Path(filename).stem
    return re.sub(r"_aug\d+$", "", stem, flags=re.IGNORECASE)

def main():
    if not IN_CSV.exists():
        print(f"❌ Input CSV not found: {IN_CSV}")
        return

    df = pd.read_csv(IN_CSV, sep=SEP)
    df["group_id"] = df["image_name"].astype(str).apply(group_id_from_name)

    # One label per group (they should be consistent by construction)
    group_df = df.groupby("group_id")["label"].first().reset_index()

    # Stratified split by label (on groups)
    train_groups, test_groups = train_test_split(
        group_df["group_id"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=group_df["label"]
    )

    train_df = df[df["group_id"].isin(train_groups)].copy()
    test_df = df[df["group_id"].isin(test_groups)].copy()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUT_DIR / "train.csv"
    test_path = OUT_DIR / "test.csv"

    # Drop helper column
    train_df.drop(columns=["group_id"], inplace=True)
    test_df.drop(columns=["group_id"], inplace=True)

    train_df.to_csv(train_path, index=False, sep=SEP, encoding="utf-8")
    test_df.to_csv(test_path, index=False, sep=SEP, encoding="utf-8")

    print("✅ Split done (grouped, no leakage).")
    print("Train rows:", len(train_df), " | Test rows:", len(test_df))
    print("Saved:", train_path)
    print("Saved:", test_path)

if __name__ == "__main__":
    main()
