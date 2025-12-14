import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
from PIL import Image
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights, ResNet50_Weights

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Alzheimer Detection Dashboard",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Alzheimer Detection Dashboard")
st.write("Upload MRI image and choose model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================
# PATH MODEL
# ======================================================
PATH_GIST = r"D:\alzheimer detection.v1i.folder\Dashboard\model\classical\gist_xgb_model.pkl"
PATH_EFF  = r"D:\alzheimer detection.v1i.folder\Dashboard\model\CNN\Efficientnet_B0_Baseline\efficientnet_b0_baseline.pkl"
PATH_LORA = r"D:\alzheimer detection.v1i.folder\Dashboard\model\CNN\Resnet50_LoRA_Fine-Tuning\resnet50_lora_best.pkl"
PATH_CNN  = r"D:\alzheimer detection.v1i.folder\Dashboard\model\CNN\cnn_scratch\cnn_scratch_best.pkl"

# ======================================================
# LOAD MODELS
# ======================================================
@st.cache_resource
def load_gist():
    return joblib.load(PATH_GIST)

@st.cache_resource
def load_effnet():
    return joblib.load(PATH_EFF)

@st.cache_resource
def load_lora():
    return joblib.load(PATH_LORA)

@st.cache_resource
def load_cnn():
    return joblib.load(PATH_CNN)

# ======================================================
# IMAGE PREPROCESS
# ======================================================
def preprocess_image(img, img_size, norm):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(norm["mean"], norm["std"])
    ])
    return transform(img).unsqueeze(0)

# ======================================================
# SIDEBAR
# ======================================================
model_choice = st.sidebar.selectbox(
    "Choose Model",
    [
        "GIST + XGBoost",
        "EfficientNet-B0",
        "ResNet50 + LoRA",
        "CNN Scratch"
    ]
)

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

# ======================================================
# PREDICTION
# ======================================================
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Running inference..."):

        # ---------------- GIST ----------------
        if model_choice == "GIST + XGBoost":
            artifact = load_gist()

            # ❗ GIST pakai fitur CSV, bukan image langsung
            st.warning("GIST model requires pre-extracted GIST features.")
            st.stop()

        # ---------------- EfficientNet ----------------
        elif model_choice == "EfficientNet-B0":
            artifact = load_effnet()

            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(
                model.classifier[1].in_features,
                artifact["num_classes"]
            )
            model.load_state_dict(artifact["model_state_dict"])
            model.to(device).eval()

            x = preprocess_image(image, artifact["img_size"], artifact["normalization"]).to(device)

        # ---------------- ResNet LoRA ----------------
        elif model_choice == "ResNet50 + LoRA":
            artifact = load_lora()

            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, artifact["num_classes"])
            model.load_state_dict(artifact["model_state_dict"])
            model.to(device).eval()

            x = preprocess_image(image, artifact["img_size"], artifact["normalization"]).to(device)

        # ---------------- CNN Scratch ----------------
        else:
            artifact = load_cnn()

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
                    return self.classifier(self.features(x))

            model = SimpleCNN(artifact["num_classes"])
            model.load_state_dict(artifact["model_state_dict"])
            model.to(device).eval()

            x = preprocess_image(image, artifact["img_size"], artifact["normalization"]).to(device)

        # ---------------- INFERENCE ----------------
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_class = artifact["class_names"][pred_idx]
        confidence = probs[pred_idx] * 100

    # ======================================================
    # OUTPUT
    # ======================================================
    st.success("Prediction Completed")
    st.metric("Predicted Class", pred_class)
    st.metric("Confidence Score", f"{confidence:.2f}%")
