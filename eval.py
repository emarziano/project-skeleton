# eval.py
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import wandb
import os

from models.custom_net import CustomNet  # importa il tuo modello
from data.prepare_tinyimagenet import prepare_tinyimagenet

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
DATASET_ROOT = "dataset"
CHECKPOINT_PATH = "checkpoints/best_model.pth"
PROJECT_NAME = "tiny-imagenet-training"

# -----------------------------
# SETUP WANDB (optional)
# -----------------------------
wandb.init(project=PROJECT_NAME, name="model-evaluation")

# -----------------------------
# 1️⃣ Dataset setup
# -----------------------------
prepare_tinyimagenet(DATASET_ROOT)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder(root=os.path.join(DATASET_ROOT, "tiny-imagenet-200", "val"), transform=transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# -----------------------------
# 2️⃣ Load model
# -----------------------------
model = CustomNet().to(DEVICE)
if os.path.exists(CHECKPOINT_PATH):
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    print(f"✅ Loaded model weights from {CHECKPOINT_PATH}")
else:
    raise FileNotFoundError(f"❌ Checkpoint not found at {CHECKPOINT_PATH}")

# -----------------------------
# 3️⃣ Evaluation loop
# -----------------------------
criterion = nn.CrossEntropyLoss()
model.eval()

total_loss, correct, total = 0.0, 0, 0

with torch.no_grad():
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

avg_loss = total_loss / len(val_loader)
accuracy = 100.0 * correct / total

# -----------------------------
# 4️⃣ Log results
# -----------------------------
print(f"📊 Evaluation results -> Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
wandb.log({"eval_loss": avg_loss, "eval_accuracy": accuracy})

wandb.finish()
