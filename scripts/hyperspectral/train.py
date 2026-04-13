# train_hyperspectral.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
import os

# ----------------------------
# 1. تحميل البيانات
# ----------------------------
df = pd.read_csv("data/hyperspectral/processed/clean.csv")

# ----------------------------
# 2. حذف الأعمدة غير مهمة
# ----------------------------
drop_cols = ['Brand', 'Acquisition', 'Class']  # أعمدة metadata
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# ----------------------------
# 3. الأعمدة الطيفية (features)
# ----------------------------
spectral_cols = [col for col in df.columns if 'nm' in col]  # 128 قيمة طيفية

# ----------------------------
# 4. تحديد Features و Target
# ----------------------------
# Classification: 0 = نقي، 1 = مغشوش
X_class = df[spectral_cols]
y_class = df['Concentration_Class'].apply(lambda x: 0 if x == 0 else 1)

# Regression: نسبة الغش
X_reg = df[spectral_cols]
y_reg = df['Concentration_Class']

# ----------------------------
# 5. تقسيم البيانات train/test
# ----------------------------
X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42)

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42)

# ----------------------------
# 6. Normalization
# ----------------------------
scaler_class = StandardScaler()
X_class_train_scaled = scaler_class.fit_transform(X_class_train)
X_class_test_scaled = scaler_class.transform(X_class_test)

scaler_reg = StandardScaler()
X_reg_train_scaled = scaler_reg.fit_transform(X_reg_train)
X_reg_test_scaled = scaler_reg.transform(X_reg_test)

# ----------------------------
# 7. تدريب النماذج
# ----------------------------
model_class = RandomForestClassifier(n_estimators=150, random_state=42)
model_class.fit(X_class_train_scaled, y_class_train)

model_reg = RandomForestRegressor(n_estimators=150, random_state=42)
model_reg.fit(X_reg_train_scaled, y_reg_train)

# ----------------------------
# 8. حفظ النماذج و scalers
# ----------------------------
os.makedirs("models/hyperspectral", exist_ok=True)

joblib.dump(model_class, "models/hyperspectral/model_class.pkl")
joblib.dump(scaler_class, "models/hyperspectral/scaler_class.pkl")
joblib.dump(model_reg, "models/hyperspectral/model_reg.pkl")
joblib.dump(scaler_reg, "models/hyperspectral/scaler_reg.pkl")

print("✅ Models trained and saved successfully")
print("Classification features shape:", X_class_train_scaled.shape)
print("Regression features shape:", X_reg_train_scaled.shape)