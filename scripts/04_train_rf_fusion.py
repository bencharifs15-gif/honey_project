import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

TRAIN_NPZ = "dataset/features/train_features.npz"
TEST_NPZ  = "dataset/features/test_features.npz"
OUT_DIR   = "models"
OUT_MODEL = os.path.join(OUT_DIR, "rf_fusion.pkl")

os.makedirs(OUT_DIR, exist_ok=True)

def load_npz(path: str):
    data = np.load(path)
    X_img = data["X_img"]   # (N, 1280)
    X_num = data["X_num"]   # (N, 3)
    y     = data["y"]       # (N,)
    return X_img, X_num, y

def fuse(X_img, X_num):
    return np.concatenate([X_img, X_num], axis=1)  # (N, 1283)

def main():
    print("Loading features...")
    X_img_tr, X_num_tr, y_tr = load_npz(TRAIN_NPZ)
    X_img_te, X_num_te, y_te = load_npz(TEST_NPZ)

    X_tr = fuse(X_img_tr, X_num_tr)
    X_te = fuse(X_img_te, X_num_te)

    print("Shapes:")
    print(" Train:", X_tr.shape, y_tr.shape)
    print(" Test :", X_te.shape, y_te.shape)

    # Random Forest
    clf = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    print("Training RandomForest...")
    clf.fit(X_tr, y_tr)

    print("Evaluating...")
    y_pred = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te)[:, 1]  # probability of class 1 (adulterated)

    acc = accuracy_score(y_te, y_pred)
    print("\nAccuracy:", round(acc, 4))

    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_te, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_te, y_pred, target_names=["pure(0)", "adulterated(1)"]))

    joblib.dump(clf, OUT_MODEL)
    print(f"\n✅ Saved model: {OUT_MODEL}")

if __name__ == "__main__":
    main()
