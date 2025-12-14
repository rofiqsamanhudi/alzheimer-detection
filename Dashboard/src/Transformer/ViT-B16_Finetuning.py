# ======================================================
# ViT B/16 Fine-Tuning (Training + Save for Streamlit)
# ======================================================

import os
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
import timm

# ======================================================
# DEVICE
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ======================================================
# PATH CONFIG
# ======================================================
BASE_DIR = "processed_dataset"

TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR   = os.path.join(BASE_DIR, "val")

SAVE_DIR = r"D:\alzheimer detection.v1i.folder\Dashboard\model\Transformer"
os.makedirs(SAVE_DIR, exist_ok=True)

# ======================================================
# DATASET CONFIG
# ======================================================
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 4
NUM_WORKERS = 2

# ======================================================
# TRANSFORM
# ======================================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# ======================================================
# DATASET & DATALOADER
# ======================================================
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
val_dataset   = datasets.ImageFolder(VAL_DIR, transform=transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

print("Classes:", train_dataset.classes)

# ======================================================
# MODEL (PRETRAINED → FINETUNE)
# ======================================================
model = timm.create_model(
    "vit_base_patch16_224",
    pretrained=True,
    num_classes=NUM_CLASSES
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

# ======================================================
# FINETUNE FUNCTION
# ======================================================
def finetune_vit(
    model,
    train_loader,
    val_loader,
    epochs=15,
    patience=3,
    lr=5e-5,
    save_dir=SAVE_DIR
):
    # Unfreeze ALL layers
    for p in model.parameters():
        p.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    trigger_times = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_model_path = os.path.join(save_dir, "vit_finetuned_best.pth")
    metrics_path    = os.path.join(save_dir, "vit_finetuned_metrics.pkl")

    for epoch in range(epochs):
        # =========================
        # TRAIN
        # =========================
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc  = correct / total

        # =========================
        # VALIDATION
        # =========================
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
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

        print(
            f"[Epoch {epoch+1}] "
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
        )

        # =========================
        # EARLY STOPPING
        # =========================
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            trigger_times = 0
            torch.save(model.state_dict(), best_model_path)
            print(">> Best model saved")
        else:
            trigger_times += 1
            print(f">> No improvement ({trigger_times}/{patience})")
            if trigger_times >= patience:
                print(">> Early stopping triggered")
                break

        # Save metrics each epoch
        with open(metrics_path, "wb") as f:
            pickle.dump({
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_accs": train_accs,
                "val_accs": val_accs
            }, f)

    print("\nFine-tuning finished")
    print("Best model:", best_model_path)
    print("Metrics   :", metrics_path)

    return best_model_path

# ======================================================
# RUN
# ======================================================
if __name__ == "__main__":
    finetune_vit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=15,
        patience=3,
        lr=5e-5
    )