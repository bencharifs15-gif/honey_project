import os
import joblib
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

MODEL_PATH = "models/rf_fusion.pkl"

def build_feature_extractor(device="cpu"):
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)
    model.classifier = torch.nn.Identity()  # output becomes 1280 feature vector
    model.eval()
    model.to(device)

    preprocess = weights.transforms()
    return model, preprocess

@torch.no_grad()
def extract_img_feat(model, preprocess, img_path: str, device="cpu"):
    img = Image.open(img_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)  # (1,3,H,W)
    feat = model(x).cpu().numpy().reshape(-1)    # (1280,)
    return feat

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run 04_train_rf_fusion.py first.")

    # ---- INPUTS (غيرهم كما تحب) ----
    img_path = input("Image path: ").strip()
    density = float(input("Density (g/ml): ").strip())
    ph      = float(input("pH: ").strip())
    flow    = float(input("Flow time (s): ").strip())
    # -------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rf = joblib.load(MODEL_PATH)
    cnn, preprocess = build_feature_extractor(device=device)

    img_feat = extract_img_feat(cnn, preprocess, img_path, device=device)  # (1280,)
    num_feat = np.array([density, ph, flow], dtype=np.float32)             # (3,)

    X = np.concatenate([img_feat, num_feat], axis=0).reshape(1, -1)        # (1,1283)

    proba_adult = float(rf.predict_proba(X)[0, 1])
    pred = int(rf.predict(X)[0])

    label = "ADULTERATED" if pred == 1 else "PURE"
    conf = proba_adult if pred == 1 else (1.0 - proba_adult)

    print("\n--- Prediction ---")
    print("Pred:", label)
    print("Confidence:", round(conf, 4))
    print("P(adulterated):", round(proba_adult, 4))

if __name__ == "__main__":
    main()
