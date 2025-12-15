import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib
from PIL import Image
from torchvision import transforms, models
from copy import deepcopy

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Alzheimer Detection Dashboard",
    layout="centered"
)

st.title("🧠 Alzheimer Detection Dashboard")
st.caption("CNN Scratch • EfficientNet-B0 • ResNet50 + LoRA")

# ======================================================
# DEVICE
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Device: **{device}**")

# ======================================================
# MODEL PATHS
# ======================================================
MODEL_PATHS = {
    "CNN - Scratch": r"D:\alzheimer detection.v1i.folder\Dashboard\src\CNN\model\cnn_scratch_best.pkl",
    "CNN - EfficientNet-B0": r"D:\alzheimer detection.v1i.folder\Dashboard\src\CNN\model\efficientnet_b0_full_finetune.pkl",
    "CNN - ResNet50 + LoRA": r"D:\alzheimer detection.v1i.folder\Dashboard\src\CNN\model\resnet50_lora_model.pkl"
}

# ======================================================
# TRANSFORM (SAMA DENGAN TRAINING)
# ======================================================
@st.cache_resource
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# ======================================================
# CNN SCRATCH MODEL
# ======================================================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# ======================================================
# LORA MODULE (HARUS IDENTIK TRAINING)
# ======================================================
class LoRALinear(nn.Module):
    def __init__(self, linear, r, alpha):
        super().__init__()
        self.linear = linear
        self.scaling = alpha / r

        self.lora_down = nn.Linear(linear.in_features, r, bias=False)
        self.lora_up   = nn.Linear(r, linear.out_features, bias=False)

        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

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
# LOAD MODEL (AUTO DETECT)
# ======================================================
@st.cache_resource
def load_model(method, path):
    artifact = joblib.load(path)
    num_classes = artifact["num_classes"]

    if method == "CNN - Scratch":
        model = SimpleCNN(num_classes)

    elif method == "CNN - EfficientNet-B0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    else:  # ResNet50 + LoRA
        base_model = models.resnet50(weights=None)
        in_features = base_model.fc.in_features
        base_model.fc = nn.Linear(in_features, num_classes)
        lora_cfg = artifact["lora"]
        model = apply_lora(base_model, lora_cfg["r"], lora_cfg["alpha"])

    model.load_state_dict(artifact["model_state_dict"])
    model.to(device)
    model.eval()

    return model, artifact["class_names"]

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
        "Select CNN Model",
        MODEL_PATHS.keys()
    )

    if st.button("🔍 Predict"):
        with st.spinner("Running model..."):
            model, class_names = load_model(
                method, MODEL_PATHS[method]
            )

            x = get_transform()(image).unsqueeze(0).to(device) # type: ignore

            with torch.no_grad():
                outputs = model(x)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                pred = np.argmax(probs)

        # ======================================================
        # RESULT
        # ======================================================
        st.success(f"🩺 Diagnosis: **{class_names[pred]}**")

        st.subheader("Prediction Probability")
        for cls, p in zip(class_names, probs):
            st.write(f"{cls}: **{p:.3f}**")
