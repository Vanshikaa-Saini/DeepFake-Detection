
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

SEED = 42
DATA_DIR = Path("data/train_10k")
MODEL_PATH = Path("models/resnet18_baseline.pth")
BATCH_SIZE = 32

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print("Device:", device)

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

# Reproduce the exact validation split used during training.
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

_, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED),
)

loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False,
)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(checkpoint["state_dict"])
model = model.to(device)
model.eval()

all_labels = []
all_predictions = []
all_probabilities = []

with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)

        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions.cpu().tolist())
        all_probabilities.extend(
            probabilities[:, 0].cpu().tolist()
        )

print("\nCheckpoint epoch:", checkpoint["epoch"])
print("Validation loss:", checkpoint["val_loss"])
print("\nConfusion matrix (rows=true, columns=predicted):")
print(confusion_matrix(all_labels, all_predictions))

print("\nClassification report:")
print(classification_report(
    all_labels,
    all_predictions,
    labels=[0, 1],
    target_names=["fake", "real"],
    zero_division=0,
))

try:
    print(
        "ROC-AUC (fake class):",
        round(roc_auc_score(
            [1 if y == 0 else 0 for y in all_labels],
            all_probabilities,
        ), 4),
    )
except ValueError as error:
    print("ROC-AUC unavailable:", error)
