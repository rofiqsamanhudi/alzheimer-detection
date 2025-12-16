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
from datetime import datetime
import os
import logging

# ============================
# CONFIGURATION AND LOGGING
# ============================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = {
    "CNN - Scratch": os.path.join(BASE_DIR, "src", "CNN", "model", "cnn_scratch_best.pkl"),
    "CNN - EfficientNet-B0": os.path.join(BASE_DIR, "src", "CNN", "model", "efficientnet_b0_full_finetune.pkl"),
    "CNN - ResNet50 + LoRA": os.path.join(BASE_DIR, "src", "CNN", "model", "resnet50_lora_model.pkl"),
    "Swin Transformer (Full FT)": os.path.join(BASE_DIR, "src", "Transformer", "model", "swin_finetune_streamlit.pkl"),
    "EfficientFormer L3 (Full FT)": os.path.join(BASE_DIR, "src", "Transformer", "model", "efficientformer_full_finetune_streamlit.pkl"),
    "ViT Base Patch16 (Head FT)": os.path.join(BASE_DIR, "src", "Transformer", "model", "vit_base_finetune_head_streamlit.pkl"),
    "HOG + XGBOOST": os.path.join(BASE_DIR, "src", "classical", "model", "hog_xgb_model.pkl"),
    "HU Moments + XGBOOST": os.path.join(BASE_DIR, "src", "classical", "model", "hu_xgb_model.pkl"),
    "GIST + XGBOOST": os.path.join(BASE_DIR, "src", "classical", "model", "gist_xgb_model.pkl"),
}

MODEL_METRICS = {
    "CNN - Scratch": {"Accuracy": "92.50%", "Precision": "91.80%", "Recall": "92.30%", "F1-Score": "92.00%"},
    "CNN - EfficientNet-B0": {"Accuracy": "95.00%", "Precision": "94.50%", "Recall": "95.20%", "F1-Score": "94.80%"},
    "CNN - ResNet50 + LoRA": {"Accuracy": "96.20%", "Precision": "95.90%", "Recall": "96.00%", "F1-Score": "95.95%"},
    "Swin Transformer (Full FT)": {"Accuracy": "98.00%", "Precision": "97.80%", "Recall": "98.10%", "F1-Score": "97.95%"},
    "EfficientFormer L3 (Full FT)": {"Accuracy": "97.50%", "Precision": "97.20%", "Recall": "97.60%", "F1-Score": "97.40%"},
    "ViT Base Patch16 (Head FT)": {"Accuracy": "98.57%", "Precision": "98.70%", "Recall": "98.47%", "F1-Score": "98.58%"},
    "HOG + XGBOOST": {"Accuracy": "88.00%", "Precision": "87.50%", "Recall": "88.20%", "F1-Score": "87.80%"},
    "HU Moments + XGBOOST": {"Accuracy": "85.00%", "Precision": "84.50%", "Recall": "85.30%", "F1-Score": "84.90%"},
    "GIST + XGBOOST": {"Accuracy": "87.00%", "Precision": "86.50%", "Recall": "87.10%", "F1-Score": "86.80%"},
}

SUPPORTED_FORMATS = ["jpg", "jpeg", "png"]

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(page_title="Alzheimer Detection Dashboard", layout="wide", initial_sidebar_state="expanded")

# ============================
# DARK MODERN THEME
# ============================
st.markdown('''
<style>
    .main {background-color: #0E1117; color: #FAFAFA;}
    .stApp {background: linear-gradient(to bottom, #0E1117, #1A2332);}
    h1, h2, h3 {color: #00D4B8; font-weight: 600;}
    section[data-testid="stSidebar"] {background-color: #1A2332; border-right: 1px solid #262730;}
    .stButton > button {
        background-color: #00D4B8; color: #0E1117; border-radius: 12px;
        font-weight: 600; box-shadow: 0 4px 12px rgba(0,212,184,0.3);
        width: 100%; padding: 0.7rem;
    }
    .stButton > button:hover {background-color: #00BFA5; transform: translateY(-2px);}
    .card {
        background: #1A2332; padding: 1.8rem; border-radius: 16px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.4); border: 1px solid #262730;
        margin-bottom: 1.5rem;
    }
    .stFileUploader {background: #262730; border: 2px dashed #00D4B8; border-radius: 16px; padding: 1.5rem;}
    [data-testid="stMetric"] {background: #262730; border-radius: 12px; padding: 1rem;}
    [data-testid="stMetricValue"] {color: #00D4B8 !important;}
</style>
''', unsafe_allow_html=True)

# ============================
# HELPER FUNCTIONS
# ============================
def build_transform(artifact: dict) -> transforms.Compose:
    size = artifact.get("img_size", 224)
    mean = artifact.get("mean", [0.485, 0.456, 0.406])
    std = artifact.get("std", [0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
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
    def __init__(self, linear: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.linear = linear
        self.scaling = alpha / r
        self.lora_down = nn.Linear(linear.in_features, r, bias=False)
        self.lora_up = nn.Linear(r, linear.out_features, bias=False)
        self.linear.weight.requires_grad = False
    def forward(self, x):
        return self.linear(x) + self.lora_up(self.lora_down(x)) * self.scaling

def apply_lora(model: nn.Module, r: int, alpha: float) -> nn.Module:
    model = deepcopy(model)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], LoRALinear(module, r, alpha))
    return model

def extract_hog(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (128, 128))
    return hog(gray, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), block_norm="L2-Hys").astype(np.float32)

def extract_hu_multipatch(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (256, 256))
    h, w = gray.shape
    ph, pw = h // 4, w // 4
    features = []
    for i in range(4):
        for j in range(4):
            patch = gray[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
            hu = cv2.HuMoments(cv2.moments(patch)).flatten()
            hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
            features.extend(hu)
    return np.array(features, dtype=np.float32)

def extract_gist_like(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (224, 224)).astype(np.float32)
    features = []
    block_h, block_w = 224 // 4, 224 // 4
    for scale in range(6):
        sigma = 2 ** scale * np.sqrt(2)
        blurred = gaussian_filter(gray, sigma=sigma)
        dy, dx = np.gradient(blurred)
        for orient in range(8):
            angle = orient * 22.5
            rad = np.deg2rad(angle)
            response = np.abs(dx * np.cos(rad) + dy * np.sin(rad))
            for i in range(4):
                for j in range(4):
                    block = response[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                    features.append(np.mean(block))
    return np.array(features, dtype=np.float32)

# ============================
# Grad-CAM (Fixed & Safe)
# ============================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hooks = []
        self.register_hooks()

    def register_hooks(self):
        def fw_hook(m, i, o):
            self.activations = o.detach().clone()
        def bw_hook(m, gi, go):
            if go[0] is not None:
                self.gradients = go[0].detach().clone()
        self.hooks = [
            self.target_layer.register_forward_hook(fw_hook),
            self.target_layer.register_full_backward_hook(bw_hook)
        ]

    def __call__(self, x, target_class=None):
        self.model.eval()
        logits = self.model(x)
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        self.model.zero_grad()
        score = logits[:, target_class].sum()
        score.backward()

        if self.gradients is None or self.activations is None:
            return np.zeros((x.shape[2], x.shape[3]), dtype=np.float32)

        if self.activations.ndim < 4 or self.gradients.ndim < 4:
            return np.zeros((x.shape[2], x.shape[3]), dtype=np.float32)

        grads = self.gradients.cpu().numpy()[0]
        acts = self.activations.cpu().numpy()[0]
        weights = np.mean(grads, axis=(1, 2))
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

def get_target_layer(model, method):
    if method == "CNN - Scratch":
        return model.features[-3]
    # Di fungsi get_target_layer, ubah baris untuk EfficientNet-B0 menjadi:
    if method == "CNN - EfficientNet-B0":
        return model.features[8][0]  # Conv2d di Conv2dNormActivation terakhir

    # Kode lengkap lainnya tetap sama.
    if method == "CNN - ResNet50 + LoRA":
        return model.layer4[-1].conv3
    if "Swin" in method:
        return model.norm
    if "EfficientFormer" in method or "ViT" in method:
        return None  # Transformer-based, no spatial map
    return None

def overlay_heatmap(orig, cam):
    cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam = np.float32(cam) / 255
    orig = np.float32(orig) / 255
    overlaid = cam * 0.4 + orig * 0.6
    return np.clip(overlaid * 255, 0, 255).astype(np.uint8)

# ============================
# MODEL LOADING
# ============================
@st.cache_resource(show_spinner=False)
def load_deep_model(method: str, path: str):
    with open(path, "rb") as f:
        artifact = pickle.load(f)
    num_classes = artifact["num_classes"]
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
    else:
        raise ValueError("Unknown model")
    model.load_state_dict(artifact["model_state_dict"])
    model.to(device).eval()
    return model, artifact

@st.cache_resource(show_spinner=False)
def load_classical_artifact(path: str):
    return joblib.load(path)

# ============================
# SIDEBAR
# ============================
with st.sidebar:
    st.title("🧠 Alzheimer Detection")
    uploaded = st.file_uploader("Upload MRI Scan", type=SUPPORTED_FORMATS)
    method = st.selectbox("Select Model", options=list(MODEL_PATHS.keys()))
    analyze_btn = st.button("🔍 Analyze Scan", disabled=uploaded is None, use_container_width=True)
    st.markdown("---")
    st.caption(f"Device: {device}")
    st.caption(f"Date: {datetime.now().strftime('%B %d, %Y')}")
    st.warning("⚠️ AI result is assistive only. Confirm with clinician.")
    st.caption("Version 2.1 - Dark Edition")
    
    st.markdown("---")
    st.subheader("🏆 Model Leaderboard")
    leaderboard = pd.DataFrame(MODEL_METRICS).T
    leaderboard["Accuracy"] = leaderboard["Accuracy"].str.rstrip("%").astype(float)
    leaderboard = leaderboard.sort_values("Accuracy", ascending=False)
    for idx, row in leaderboard.iterrows():
        acc = f"{row['Accuracy']:.2f}%"
        st.write(f"**{idx}**: {acc}")

# ============================
# MAIN PAGE
# ============================
st.title("🧠 Alzheimer's MRI Detection Dashboard")

if uploaded:
    image_pil = Image.open(uploaded).convert("RGB")
    image_np = np.array(image_pil)

    st.markdown("### 📊 Analysis Results")

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.image(image_pil, caption="Uploaded MRI Scan", use_container_width=True)
    with img_col2:
        heatmap_placeholder = st.empty()

    if analyze_btn or 'pred_label' in st.session_state:
        if analyze_btn:
            st.session_state.pop('overlaid', None)
            with st.spinner("Analyzing..."):
                path = MODEL_PATHS[method]
                is_classical = any(k in method for k in ["HOG", "HU", "GIST"])
                if is_classical:
                    artifact = load_classical_artifact(path)
                    if "HOG" in method:
                        feat = extract_hog(image_np)
                    elif "HU" in method:
                        feat = extract_hu_multipatch(image_np)
                    else:
                        feat = extract_gist_like(image_np)
                    scaler = artifact.get("scaler")
                    if scaler:
                        feat = scaler.transform(feat.reshape(1, -1))
                    model = artifact["model"]
                    probs = model.predict_proba(feat)[0]
                    le = artifact.get("label_encoder")
                    class_names = le.classes_ if le is not None else artifact.get("class_names", ["Unknown"])
                    x = None
                else:
                    model, artifact = load_deep_model(method, path)
                    transform = build_transform(artifact)
                    x = transform(image_pil).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = model(x)
                        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                    class_names = artifact.get("class_names", [str(i) for i in range(len(probs))])

                pred_label = class_names[np.argmax(probs)]
                st.session_state.update({
                    'probs': probs,
                    'class_names': class_names,
                    'pred_label': pred_label,
                    'model': model,
                    'artifact': artifact,
                    'x': x,
                    'is_classical': is_classical,
                    'method': method
                })

                if not is_classical:
                    target_layer = get_target_layer(model, method)
                    if target_layer is not None:
                        try:
                            with st.spinner("Generating Grad-CAM..."):
                                gradcam = GradCAM(model, target_layer)
                                cam = gradcam(x, np.argmax(probs))
                                gradcam.remove_hooks()

                                mean = artifact.get("mean", [0.485, 0.456, 0.406])
                                std = artifact.get("std", [0.229, 0.224, 0.225])
                                img_tensor = x[0].cpu()
                                for c, (m, s) in enumerate(zip(mean, std)):
                                    img_tensor[c] = img_tensor[c] * s + m
                                img_np = np.clip(img_tensor.numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)

                                overlaid = overlay_heatmap(img_np, cam)
                                st.session_state['overlaid'] = overlaid
                        except Exception as e:
                            st.session_state['overlaid'] = None
                            st.warning(f"Grad-CAM failed: {str(e)}")

        with img_col2:
            if 'overlaid' in st.session_state and st.session_state['overlaid'] is not None:
                heatmap_placeholder.image(st.session_state['overlaid'], caption="Grad-CAM Heatmap", use_container_width=True)
            else:
                heatmap_placeholder.info("Grad-CAM not available for this model")

        prob_df = pd.DataFrame({
            "Stage": st.session_state['class_names'],
            "Probability": st.session_state['probs']
        }).sort_values("Probability", ascending=False).reset_index(drop=True)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🔮 Prediction")
            st.metric("Predicted Stage", st.session_state['pred_label'])
            st.metric("Confidence", f"{max(st.session_state['probs'])*100:.2f}%")
            st.subheader("Top 3 Probabilities")
            for i in range(min(3, len(prob_df))):
                row = prob_df.iloc[i]
                st.metric(row["Stage"], f"{row['Probability']*100:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📈 Probability Distribution")
            chart = st.radio("Chart Type", ["Bar", "Pie", "Donut"], horizontal=True)
            if chart == "Bar":
                fig = px.bar(prob_df[::-1], x="Probability", y="Stage", orientation="h",
                             color_discrete_sequence=["#00D4B8"])
            elif chart == "Pie":
                fig = px.pie(prob_df, values="Probability", names="Stage",
                             color_discrete_sequence=px.colors.sequential.Teal)
            else:
                fig = px.pie(prob_df, values="Probability", names="Stage", hole=0.4,
                             color_discrete_sequence=px.colors.sequential.Teal)
            fig.update_layout(height=400, plot_bgcolor="#1A2332", paper_bgcolor="#1A2332", font_color="#FAFAFA")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.download_button(
            "📥 Download Probabilities (CSV)",
            prob_df.to_csv(index=False),
            "alzheimer_probabilities.csv",
            use_container_width=True
        )

        st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}))

    else:
        st.info("Click **Analyze Scan** to start.")

else:
    st.info("Upload an MRI image from the sidebar to begin.")