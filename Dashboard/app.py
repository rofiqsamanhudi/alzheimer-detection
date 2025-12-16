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
# PAGE CONFIG & DARK THEME
# ============================
st.set_page_config(page_title="Alzheimer Detection Dashboard", layout="wide", initial_sidebar_state="expanded")

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
        nn.init.kaiming_uniform_(self.lora_down.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

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
    return hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys").astype(np.float32)

def extract_hu_multipatch(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (256, 256))
    h, w = gray.shape
    ph, pw = h // 4, w // 4
    features = []
    for i in range(4):
        for j in range(4):
            patch = gray[i * ph:(i + 1) * ph, j * pw:(j + 1) * pw]
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
                    block = response[i * block_h:(i + 1) * block_h, j * block_w:(j + 1) * block_w]
                    features.append(np.mean(block))
    return np.array(features, dtype=np.float32)

# ============================
# HEATMAP GENERATION
# ============================
def clear_all_backward_hooks(model: nn.Module):
    for module in model.modules():
        if hasattr(module, '_backward_hooks'):
            module._backward_hooks.clear()
        if hasattr(module, '_full_backward_hooks'):
            module._full_backward_hooks.clear()
        if hasattr(module, '_full_backward_pre_hooks'):
            module._full_backward_pre_hooks.clear()

def generate_heatmap(model, x, method, class_idx=None):
    model.eval()
    x = x.clone().detach().requires_grad_(True)

    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def full_backward_hook(module, grad_input, grad_output):
        if len(grad_output) > 0 and grad_output[0] is not None:
            gradients.append(grad_output[0].detach())

    def regular_backward_hook(module, grad_input, grad_output):
        if len(grad_output) > 0 and grad_output[0] is not None:
            gradients.append(grad_output[0].detach())

    # Target layer selection
    if "EfficientNet" in method:
        target_layer = model.features[6]
        hook_type = "full"
    elif "ResNet50" in method:
        target_layer = model.layer4
        hook_type = "full"
    elif "Scratch" in method:
        target_layer = model.features[-1]
        hook_type = "regular"
    elif "ViT" in method:
        target_layer = model.blocks[-1].norm1
        hook_type = "full"
    elif "Swin" in method:
        target_layer = model.norm
        hook_type = "full"
    elif "EfficientFormer" in method:
        last_stage = model.stages[-1]
        if hasattr(last_stage, 'blocks') and len(last_stage.blocks) > 0:
            last_block = last_stage.blocks[-1]
            if hasattr(last_block, 'ffn') and hasattr(last_block.ffn, 'fc2'):
                target_layer = last_block.ffn.fc2
            elif hasattr(last_block, 'mlp'):
                target_layer = last_block.mlp.fc2 if hasattr(last_block.mlp, 'fc2') else last_block.mlp
            elif hasattr(last_block, 'conv'):
                target_layer = last_block.conv
            else:
                target_layer = last_block
        else:
            target_layer = last_stage
        hook_type = "full"
    else:
        return None

    clear_all_backward_hooks(model)
    fwd_handle = target_layer.register_forward_hook(forward_hook)

    try:
        if hook_type == "regular":
            bwd_handle = target_layer.register_backward_hook(regular_backward_hook)
        else:
            bwd_handle = target_layer.register_full_backward_hook(full_backward_hook)
    except RuntimeError as e:
        if "both regular" in str(e).lower():
            clear_all_backward_hooks(model)
            bwd_handle = target_layer.register_backward_hook(regular_backward_hook)
        else:
            raise e

    try:
        logits = model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        model.zero_grad()
        logits[0, class_idx].backward()

        if len(activations) == 0 or len(gradients) == 0:
            logging.warning(f"No activation/gradient captured for {method}")
            return None

        act = activations[0]
        grad = gradients[0]

        # Handle transformer-style activations
        if any(t in method for t in ["ViT", "Swin", "EfficientFormer"]):
            if act.dim() == 3:
                B, seq_len, C = act.shape
                act = act[:, 1:, :]
                grad = grad[:, 1:, :]
                patch_size = int((seq_len - 1) ** 0.5)
                if patch_size * patch_size != seq_len - 1:
                    return None
                act = act.reshape(B, patch_size, patch_size, C).permute(0, 3, 1, 2)
                grad = grad.reshape(B, patch_size, patch_size, C).permute(0, 3, 1, 2)

        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * act).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam[0].cpu().numpy()

        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    except Exception as e:
        logging.error(f"Heatmap generation failed for {method}: {e}")
        return None
    finally:
        fwd_handle.remove()
        bwd_handle.remove()

def overlay_heatmap(original_image: np.ndarray, cam: np.ndarray) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET) # pyright: ignore[reportArgumentType, reportCallIssue]
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = heatmap * 0.4 + original_image * 0.6
    return np.clip(overlay, 0, 255).astype(np.uint8)

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
# SIDEBAR & MAIN INTERFACE
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
    st.markdown("---")
    st.subheader("🏆 Model Leaderboard")
    leaderboard = pd.DataFrame(MODEL_METRICS).T
    leaderboard["Accuracy"] = leaderboard["Accuracy"].str.rstrip("%").astype(float)
    leaderboard = leaderboard.sort_values("Accuracy", ascending=False)
    for idx, row in leaderboard.iterrows():
        st.write(f"**{idx}**: {row['Accuracy']:.2f}%")

st.title("Alzheimer's MRI Detection Dashboard")

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
            st.session_state.clear()
            with st.spinner("Analyzing image and generating explanation..."):
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
                    class_names = artifact.get("class_names", ["Non-Demented", "Mild", "Moderate", "Very Mild"])
                    overlaid = None
                else:
                    model, artifact = load_deep_model(method, path)
                    transform = build_transform(artifact)
                    x = transform(image_pil).unsqueeze(0).to(device) # pyright: ignore[reportAttributeAccessIssue]
                    with torch.no_grad():
                        logits = model(x)
                        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                    class_names = artifact.get("class_names", [f"Class {i}" for i in range(len(probs))])
                    cam = generate_heatmap(model, x, method, class_idx=np.argmax(probs))
                    overlaid = overlay_heatmap(image_np, cam) if cam is not None else None

                pred_label = class_names[np.argmax(probs)]
                st.session_state.update({
                    'probs': probs.tolist(),
                    'class_names': class_names,
                    'pred_label': pred_label,
                    'is_classical': is_classical,
                    'method': method,
                    'overlaid': overlaid
                })

        # Display heatmap
        with img_col2:
            if st.session_state.get('overlaid') is not None:
                heatmap_placeholder.image(
                    st.session_state['overlaid'],
                    caption="🔥 Model Explanation Heatmap (Grad-CAM)",
                    use_container_width=True
                )
                st.caption("Red/orange areas indicate regions most influential to the model's prediction.")
            else:
                if st.session_state.get('is_classical'):
                    heatmap_placeholder.info("ℹ️ Classical models do not support visual explanation heatmaps.")
                else:
                    heatmap_placeholder.warning("⚠️ Heatmap generation failed for this model. Prediction is still valid.")

        # Results layout
        prob_df = pd.DataFrame({
            "Stage": st.session_state['class_names'],
            "Probability": st.session_state['probs']
        }).sort_values("Probability", ascending=False).reset_index(drop=True)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🔮 Prediction")
            st.metric("Predicted Stage", st.session_state['pred_label'])
            st.metric("Confidence", f"{max(st.session_state['probs']) * 100:.2f}%")
            st.subheader("Top 3 Probabilities")
            for i in range(min(3, len(prob_df))):
                row = prob_df.iloc[i]
                st.metric(row["Stage"], f"{row['Probability'] * 100:.2f}%")
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