# ======================================================
# EfficientNet-B0 Baseline (Training + Save for Streamlit)
# ======================================================

import os
import torch
import joblib
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights
from torch.utils.data import DataLoader

# ======================================================
# DEVICE
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", device)

# ======================================================
# PATH CONFIG
# ======================================================
BASE_DIR = "processed_dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR   = os.path.join(BASE_DIR, "test")   # dipakai sebagai validation

SAVE_DIR = r"D:\alzheimer detection.v1i.folder\Dashboard\model\CNN"
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_PATH = os.path.join(SAVE_DIR, "efficientnet_b0_baseline.pkl")

# ======================================================
# CONFIG
# ======================================================
NUM_CLASSES = 4
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
PATIENCE = 3
LR = 1e-4

# ======================================================
# TRANSFORMS
# ======================================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ======================================================
# DATASET & DATALOADER
# ======================================================
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
val_dataset   = datasets.ImageFolder(VAL_DIR, transform=transform)

class_names = train_dataset.classes

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ======================================================
# MODEL INITIALIZATION
# ======================================================
weights = EfficientNet_B0_Weights.DEFAULT
model = models.efficientnet_b0(weights=weights)

in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

model = model.to(device)

# ======================================================
# LOSS & OPTIMIZER
# ======================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler()

# ======================================================
# TRAINING LOOP
# ======================================================
best_val_acc = 0.0
patience_counter = 0

train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(EPOCHS):
    # ---------------- TRAIN ----------------
    model.train()
    running_loss, correct, total = 0, 0, 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    # ---------------- VALIDATION ----------------
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = outputs.max(1)
            val_total += labels.size(0)
            val_correct += preds.eq(labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = val_correct / val_total

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"[Epoch {epoch+1}] "
          f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    # ---------------- EARLY STOP + SAVE BEST ----------------
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0

        artifact = {
            "model_state_dict": model.state_dict(),
            "architecture": "EfficientNet-B0",
            "num_classes": NUM_CLASSES,
            "class_names": class_names,
            "img_size": IMG_SIZE,
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            },
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accs": train_accs,
            "val_accs": val_accs
        }

        joblib.dump(artifact, MODEL_PATH)
        print(f"Best model saved | Val Acc: {best_val_acc:.4f}")

    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

print("\n[SUCCESS] EfficientNet-B0 baseline model ready for Streamlit")
print("Saved at:", MODEL_PATH)
