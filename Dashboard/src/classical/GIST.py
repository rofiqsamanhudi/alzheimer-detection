# ======================================================
# GIST XGBoost Model - FINAL STABLE VERSION
# ======================================================

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, confusion_matrix

from xgboost import XGBClassifier

# ======================================================
# PATH
# ======================================================
FEATURE_PATH = r"Fitur_Ekstraksi_Klasik\features_extracted\features_gist_like_multiblock.csv"
SAVE_DIR = r"D:\alzheimer detection.v1i.folder\Dashboard\model\classical"
os.makedirs(SAVE_DIR, exist_ok=True)
MODEL_PATH = os.path.join(SAVE_DIR, "gist_xgb_model_final.pkl")

# ======================================================
# LOAD DATA
# ======================================================
df = pd.read_csv(FEATURE_PATH)

feature_cols = [c for c in df.columns if c.startswith("f")]
X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
y = df["label"]

print("\nClass distribution:")
print(y.value_counts())

# ======================================================
# LABEL ENCODING
# ======================================================
label_encoder = LabelEncoder()
y_enc = label_encoder.fit_transform(y)

# ======================================================
# FEATURE FILTERING (SAFE)
# ======================================================
selector = VarianceThreshold(threshold=1e-3)
X_sel = selector.fit_transform(X)

print(f"Features before : {X.shape[1]}")
print(f"Features after  : {X_sel.shape[1]}")

# ======================================================
# SCALING
# ======================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sel)

# ======================================================
# SPLIT
# ======================================================
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_scaled, y_enc,
    test_size=0.2,
    stratify=y_enc,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=0.25,
    stratify=y_trainval,
    random_state=42
)

# ======================================================
# CLASS IMBALANCE
# ======================================================
sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

# ======================================================
# XGBOOST — STABLE CONFIG
# ======================================================
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=3,              # 🔥 kunci stabilitas
    min_child_weight=5,       # 🔥 cegah noise split
    learning_rate=0.05,
    subsample=0.75,
    colsample_bytree=0.6,
    reg_alpha=0.8,
    reg_lambda=1.2,
    max_delta_step=1,         # 🔥 stabilkan probabilitas
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42
)

xgb_model.fit(
    X_train,
    y_train,
    sample_weight=sample_weights,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# ======================================================
# EVALUATION
# ======================================================
y_val_pred = xgb_model.predict(X_val)

print("\nValidation Classification Report:")
print(classification_report(
    y_val, y_val_pred,
    target_names=label_encoder.classes_
))

print("Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))

# ======================================================
# SAVE ARTIFACT
# ======================================================
artifact = {
    "model": xgb_model,
    "scaler": scaler,
    "selector": selector,
    "label_encoder": label_encoder,
    "feature_type": "GIST-like Multiblock (Stable Final)",
    "num_features": X_sel.shape[1],
    "class_names": list(label_encoder.classes_)
}

joblib.dump(artifact, MODEL_PATH)

print("\n[SUCCESS] FINAL GIST model saved")
print("Path :", MODEL_PATH)
