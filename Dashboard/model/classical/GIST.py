# ======================================================
# GIST XGBoost Model Training & Saving
# ======================================================

import os
import joblib
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# ======================================================
# PATH CONFIGURATION
# ======================================================
FEATURE_PATH = r"Fitur_Ekstraksi_Klasik\features_extracted\features_gist_like_multiblock.csv"

SAVE_DIR = r"D:\alzheimer detection.v1i.folder\Dashboard\model\classical"
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_PATH = os.path.join(SAVE_DIR, "gist_xgb_model.pkl")


# ======================================================
# LOAD & PREPARE DATA
# ======================================================
df = pd.read_csv(FEATURE_PATH)

# Ambil hanya kolom fitur GIST (f0, f1, ..., fn)
feature_cols = [c for c in df.columns if c.startswith("f")]

X = df[feature_cols]
X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

y = df["label"]


# ======================================================
# LABEL ENCODING & SCALING
# ======================================================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ======================================================
# DATA SPLIT (60% TRAIN | 20% VAL | 20% TEST)
# ======================================================
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.2,
    stratify=y_encoded,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval,
    y_trainval,
    test_size=0.25,
    stratify=y_trainval,
    random_state=42
)


# ======================================================
# TRAIN XGBOOST CLASSIFIER (GIST FEATURES)
# ======================================================
xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42
)

xgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)


# ======================================================
# SAVE MODEL ARTIFACT (STREAMLIT READY)
# ======================================================
artifact = {
    "model": xgb_model,
    "scaler": scaler,
    "label_encoder": label_encoder,
    "feature_type": "GIST-like Multiblock",
    "num_features": X.shape[1],
    "class_names": list(label_encoder.classes_)
}

joblib.dump(artifact, MODEL_PATH)


# ======================================================
# LOG OUTPUT
# ======================================================
print("\n[SUCCESS] GIST XGBoost model berhasil disimpan!")
print(f"Path           : {MODEL_PATH}")
print(f"Jumlah fitur   : {X.shape[1]}")
print(f"Nama kelas     : {artifact['class_names']}")