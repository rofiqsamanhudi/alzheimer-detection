import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib
import pickle
from PIL import Image
from torchvision import transforms, models
from timm import create_model
from copy import deepcopy
import cv2
from skimage.feature import hog
from scipy.ndimage import gaussian_filter
import plotly.express as px
import pandas as pd

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Alzheimer Detection Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# CUSTOM CSS FOR BETTER UI
# ======================================================
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { background-color: #007bff; color: white; border-radius: 5px; }
    .stSelectbox { background-color: #ffffff; border: 1px solid #ced4da; border-radius: 5px; }
    .stFileUploader { border: 1px dashed #ced4da; border-radius: 5px; padding: 10px; }
    .stSpinner { color: #007bff; }
    .stSuccess { background-color: #d4edda; border-color: #c3e6cb; color: #155724; }
    h1, h2, h3 { color: #343a40; }
    .sidebar .sidebar-content { background-color: #e9ecef; }
    </style>
""", unsafe_allow_html=True)

# ======================================================
# SIDEBAR
# ======================================================
with st.sidebar:
    st.header("About This Dashboard")
    st.markdown("""
    This academic dashboard demonstrates various machine learning and deep learning models for Alzheimer's disease detection from MRI images.
    
    **Key Features:**
    - Supports CNN, Transformer, and Classical ML models.
    - Real-time inference on uploaded images.
    - Visual probability distributions.
    
    **Dataset Assumption:** Models trained on Alzheimer's MRI dataset (e.g., classes: MildDemented, ModerateDemented, NonDemented, VeryMildDemented).
    
    For educational purposes only. Not for clinical use.
    """)
    
    st.header("Model Descriptions")
    model_descriptions = {
        "CNN - Scratch": "A simple convolutional neural network built from scratch with basic layers.",
        "CNN - EfficientNet-B0": "EfficientNet-B0 architecture with full fine-tuning for efficient performance.",
        "CNN - ResNet50 + LoRA": "ResNet50 with Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning.",
        "Swin Transformer (Full FT)": "Swin Transformer with full fine-tuning, leveraging shifted windows for hierarchical vision.",
        "EfficientFormer L3 (Full FT)": "EfficientFormer L3 model, a hybrid vision transformer with full fine-tuning.",
        "ViT Base Patch16 (Head FT)": "Vision Transformer base with patch size 16, fine-tuned on the classification head.",
        "HOG + XGBOOST": "Histogram of Oriented Gradients features combined with XGBoost classifier.",
        "HU Moments + XGBOOST": "Hu Moments (shape descriptors) extracted from image patches with XGBoost.",
        "GIST + XGBOOST": "GIST global image descriptors with XGBoost for scene understanding."
    }

# ======================================================
# MAIN CONTENT
# ======================================================
st.title("🧠 Alzheimer Detection Dashboard")
st.caption("An Academic Tool for MRI-Based Alzheimer's Classification Using CNN, Transformers, and Classical Models")

# Device Info
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.info(f"Running on device: **{device}**")

# ======================================================
# MODEL PATHS (UNCHANGED)
# ======================================================
MODEL_PATHS = {
    # CNN
    "CNN - Scratch": r"D:\alzheimer detection.v1i.folder\Dashboard\src\CNN\model\cnn_scratch_best.pkl",
    "CNN - EfficientNet-B0": r"D:\alzheimer detection.v1i.folder\Dashboard\src\CNN\model\efficientnet_b0_full_finetune.pkl",
    "CNN - ResNet50 + LoRA": r"D:\alzheimer detection.v1i.folder\Dashboard\src\CNN\model\resnet50_lora_model.pkl",
   
    # Transformer
    "Swin Transformer (Full FT)": r"D:\alzheimer detection.v1i.folder\Dashboard\src\Transformer\model\swin_finetune_streamlit.pkl",
    "EfficientFormer L3 (Full FT)": r"D:\alzheimer detection.v1i.folder\Dashboard\src\Transformer\model\efficientformer_full_finetune_streamlit.pkl",
    "ViT Base Patch16 (Head FT)": r"D:\alzheimer detection.v1i.folder\Dashboard\src\Transformer\model\vit_base_finetune_head_streamlit.pkl",
   
    # Classical
    "HOG + XGBOOST": r"D:\alzheimer detection.v1i.folder\Dashboard\src\classical\model\hog_xgb_model.pkl",
    "HU Moments + XGBOOST": r"D:\alzheimer detection.v1i.folder\Dashboard\src\classical\model\hu_xgb_model.pkl",
    "GIST + XGBOOST": r"D:\alzheimer detection.v1i.folder\Dashboard\src\classical\model\gist_xgb_model.pkl",
}

# ======================================================
# BUILD TRANSFORM UNTUK DEEP LEARNING (UNCHANGED)
# ======================================================
def build_transform(artifact):
    size = artifact.get("img_size", 224)
    if "normalization" in artifact:
        mean = artifact["normalization"].get("mean", [0.485, 0.456, 0.406])
        std = artifact["normalization"].get("std", [0.229, 0.224, 0.225])
    else:
        mean = artifact.get("mean", [0.485, 0.456, 0.406])
        std = artifact.get("std", [0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

# ======================================================
# CNN & LoRA (UNCHANGED)
# ======================================================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))
class LoRALinear(nn.Module):
    def __init__(self, linear, r, alpha):
        super().__init__()
        self.linear = linear
        self.scaling = alpha / r
        self.lora_down = nn.Linear(linear.in_features, r, bias=False)
        self.lora_up = nn.Linear(r, linear.out_features, bias=False)
        self.linear.weight.requires_grad = False
    def forward(self, x):
        return self.linear(x) + self.lora_up(self.lora_down(x)) * self.scaling
def apply_lora(model, r, alpha):
    model = deepcopy(model)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], LoRALinear(module, r, alpha))
    return model

# ======================================================
# CLASSICAL FEATURE EXTRACTORS (UNCHANGED)
# ======================================================
# HOG
IMG_SIZE_HOG = 128
HOG_PARAMS = dict(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys")
def extract_hog(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE_HOG, IMG_SIZE_HOG))
    feat = hog(
        gray,
        orientations=HOG_PARAMS["orientations"],
        pixels_per_cell=HOG_PARAMS["pixels_per_cell"],
        cells_per_block=HOG_PARAMS["cells_per_block"],
        block_norm=HOG_PARAMS["block_norm"],
        feature_vector=True
    )
    return feat.astype(np.float32)
# HU Moments
IMG_SIZE_HU = 256
GRID_HU = 4
def extract_hu_multipatch(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE_HU, IMG_SIZE_HU))
    h, w = gray.shape
    ph, pw = h // GRID_HU, w // GRID_HU
    features = []
    for i in range(GRID_HU):
        for j in range(GRID_HU):
            patch = gray[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
            hu = cv2.HuMoments(cv2.moments(patch)).flatten()
            hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
            features.extend(hu)
    return np.array(features, dtype=np.float32)
# GIST (DIPERBAIKI: 6 scales × 8 orientations × 16 blocks = 768 dimensi)
IMG_SIZE_GIST = 224
GRID_GIST = 4 # 4x4 = 16 blocks
NUM_SCALES = 6
NUM_ORIENTATIONS_PER_SCALE = 8
def extract_gist_like(image: np.ndarray) -> np.ndarray:  # Renamed to match pickle reference
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE_GIST, IMG_SIZE_GIST))
    gray = gray.astype(np.float32)
    features = []
    block_h = IMG_SIZE_GIST // GRID_GIST
    block_w = IMG_SIZE_GIST // GRID_GIST
    for scale in range(NUM_SCALES):
        sigma = 2 ** scale * np.sqrt(2) # standard scaling
        blurred = gaussian_filter(gray, sigma=sigma)
        for orient in range(NUM_ORIENTATIONS_PER_SCALE):
            angle = orient * (180.0 / NUM_ORIENTATIONS_PER_SCALE)
            rad = np.deg2rad(angle)
            # Aproksimasi filter orientasi menggunakan gradient
            dy, dx = np.gradient(blurred)
            oriented_response = np.abs(dx * np.cos(rad) + dy * np.sin(rad))
            # Ekstrak mean dari setiap block 4x4
            for i in range(GRID_GIST):
                for j in range(GRID_GIST):
                    block = oriented_response[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                    features.append(np.mean(block))
    return np.array(features, dtype=np.float32)

# ======================================================
# LOAD DEEP MODEL (UNCHANGED)
# ======================================================
@st.cache_resource
def load_deep_model(method, path):
    try:
        artifact = joblib.load(path)
    except:
        with open(path, "rb") as f:
            artifact = pickle.load(f)
   
    num_classes = artifact["num_classes"]
    model = None
    if method == "CNN - Scratch":
        model = SimpleCNN(num_classes)
    elif method == "CNN - EfficientNet-B0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif method == "CNN - ResNet50 + LoRA":
        base = models.resnet50(weights=None)
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        model = apply_lora(base, artifact["lora"]["r"], artifact["lora"]["alpha"])
    elif "Swin" in method:
        model = models.swin_t(weights=None)
        model.head = nn.Linear(model.head.in_features, num_classes)
    elif "EfficientFormer" in method:
        model = create_model("efficientformer_l3", pretrained=False, num_classes=num_classes)
    elif "ViT" in method:
        model = create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)
    if model is None:
        raise ValueError(f"Model tidak dikenali: {method}")
    model.load_state_dict(artifact["model_state_dict"])
    model.to(device).eval()
    return model, artifact

# ======================================================
# LOAD CLASSICAL ARTIFACT (UNCHANGED)
# ======================================================
@st.cache_resource
def load_classical_artifact(path):
    return joblib.load(path)

# ======================================================
# UI IMPROVEMENTS
# ======================================================
st.header("1. Upload MRI Image")
uploaded = st.file_uploader("Choose an MRI image file", type=["jpg", "jpeg", "png"], help="Upload a brain MRI scan for analysis.")

if uploaded:
    image_pil = Image.open(uploaded).convert("RGB")
    image_np = np.array(image_pil)
    
    # Display uploaded image in a column layout
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Uploaded Image")
        st.image(image_pil, caption="Original MRI Scan", use_column_width=True)
    
    with col2:
        st.subheader("Image Details")
        st.write(f"Dimensions: {image_pil.width} x {image_pil.height}")
        st.write(f"Format: {image_pil.format}")
        st.write(f"Mode: {image_pil.mode}")

    st.header("2. Select Model")
    method = st.selectbox("Choose a Model for Prediction", options=list(MODEL_PATHS.keys()), help="Select from CNN, Transformer, or Classical models.")
    
    # Display model description
    with st.expander("Model Description"):
        st.markdown(model_descriptions.get(method, "No description available."))

    if st.button("🔍 Run Prediction", help="Click to perform inference using the selected model."):
        with st.spinner("Running inference... Please wait."):
            path = MODEL_PATHS[method]
            is_classical = "HOG" in method or "HU" in method or "GIST" in method
            if is_classical:
                # ------------------- CLASSICAL -------------------
                artifact = load_classical_artifact(path)
                if "HOG" in method:
                    feat = extract_hog(image_np)
                elif "HU" in method:
                    feat = extract_hu_multipatch(image_np)
                elif "GIST" in method:
                    feat = extract_gist_like(image_np)  # Updated call
                else:
                    st.error("Extractor tidak ditemukan!")
                    st.stop()
                scaler = artifact["scaler"]
                if feat.shape[0] != scaler.mean_.shape[0]:
                    st.error(f"Dimensi fitur tidak cocok: {feat.shape[0]} vs {scaler.mean_.shape[0]}")
                    st.stop()
                feat_scaled = scaler.transform(feat.reshape(1, -1))
                model = artifact["model"]
                probs = model.predict_proba(feat_scaled)[0]
                pred_idx = np.argmax(probs)
                le = artifact["label_encoder"]
                pred_label = le.inverse_transform([pred_idx])[0]
                class_names = le.classes_
            else:
                # ------------------- DEEP LEARNING -------------------
                model, artifact = load_deep_model(method, path)
                transform = build_transform(artifact)
                x = transform(image_pil).unsqueeze(0).to(device) # pyright: ignore[reportAttributeAccessIssue]
                with torch.no_grad():
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
               
                pred_idx = np.argmax(probs)
                pred_label = artifact["class_names"][pred_idx]
                class_names = artifact["class_names"]
            # ======================================================
            # HASIL PREDIKSI (ENHANCED UI)
            # ======================================================
            st.header("3. Prediction Results")
            st.success(f"🩺 Predicted Diagnosis: **{pred_label}**")
            
            st.subheader("Class Probabilities")
            # Create DataFrame for probabilities
            prob_df = pd.DataFrame({
                "Class": class_names,
                "Probability": probs
            }).sort_values(by="Probability", ascending=False)
            
            # Display table
            st.table(prob_df.style.format({"Probability": "{:.3f}"}))
            
            # Visualize with Plotly bar chart
            fig = px.bar(prob_df, x="Class", y="Probability", 
                         color="Class", 
                         range_y=[0, 1],
                         title="Probability Distribution",
                         labels={"Probability": "Probability Score"},
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(xaxis_title="Alzheimer's Stage", yaxis_title="Probability", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional academic note
            with st.expander("Interpretation Notes"):
                st.markdown("""
                - **NonDemented**: No signs of Alzheimer's.
                - **VeryMildDemented**: Early-stage changes.
                - **MildDemented**: Noticeable symptoms.
                - **ModerateDemented**: Advanced stage.
                
                Probabilities indicate model confidence. Consult a medical professional for accurate diagnosis.
                """)

else:
    st.warning("Please upload an MRI image to begin.")