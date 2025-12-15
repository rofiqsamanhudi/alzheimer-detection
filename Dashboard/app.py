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

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Alzheimer Detection Dashboard",
    layout="centered"
)

st.title("🧠 Alzheimer Detection Dashboard")
st.caption("CNN & Transformer Models")

# ======================================================
# DEVICE
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Device: **{device}**")

# ======================================================
# MODEL PATHS
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
}

# ======================================================
# BUILD TRANSFORM FROM ARTIFACT
# ======================================================
def build_transform(artifact):
    mean = artifact["normalization"]["mean"]
    std  = artifact["normalization"]["std"]
    size = artifact["img_size"]

    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

# ======================================================
# CNN SCRATCH
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

# ======================================================
# LORA MODULE
# ======================================================
class LoRALinear(nn.Module):
    def __init__(self, linear, r, alpha):
        super().__init__()
        self.linear = linear
        self.scaling = alpha / r
        self.lora_down = nn.Linear(linear.in_features, r, bias=False)
        self.lora_up   = nn.Linear(r, linear.out_features, bias=False)
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
# LOAD MODEL (CNN + TRANSFORMER)
# ======================================================
@st.cache_resource
def load_model(method, path):
    try:
        artifact = joblib.load(path)
    except:
        with open(path, "rb") as f:
            artifact = pickle.load(f)

    num_classes = artifact["num_classes"]
    model = None  # 🔥 FIX: deklarasi awal

    # ================= CNN =================
    if method == "CNN - Scratch":
        model = SimpleCNN(num_classes)

    elif method == "CNN - EfficientNet-B0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, num_classes
        )

    elif method == "CNN - ResNet50 + LoRA":
        base = models.resnet50(weights=None)
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        model = apply_lora(
            base,
            artifact["lora"]["r"],
            artifact["lora"]["alpha"]
        )

    # ================= TRANSFORMER =================
    elif "Swin" in method:
        model = models.swin_t(weights=None)
        model.head = nn.Linear(model.head.in_features, num_classes)

    elif "EfficientFormer" in method:
        model = create_model(
            "efficientformer_l3",
            pretrained=False,
            num_classes=num_classes
        )

    elif "ViT" in method:
        model = create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=num_classes
        )

    # ================= SAFETY CHECK =================
    if model is None:
        raise ValueError(f"Model method tidak dikenali: {method}")

    model.load_state_dict(artifact["model_state_dict"])
    model.to(device).eval()

    return model, artifact

# ======================================================
# UI
# ======================================================
uploaded = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded MRI", width=300)

    method = st.selectbox(
        "Select Model",
        MODEL_PATHS.keys()
    )

    if st.button("🔍 Predict"):
        with st.spinner("Running inference..."):
            model, artifact = load_model(
                method, MODEL_PATHS[method]
            )

            transform = build_transform(artifact)
            x = transform(image).unsqueeze(0).to(device) # pyright: ignore[reportAttributeAccessIssue]

            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                pred = np.argmax(probs)

        # ======================================================
        # RESULT
        # ======================================================
        st.success(
            f"🩺 Diagnosis: **{artifact['class_names'][pred]}**"
        )

        st.subheader("Prediction Probability")
        for cls, p in zip(artifact["class_names"], probs):
            st.write(f"{cls}: **{p:.3f}**")
