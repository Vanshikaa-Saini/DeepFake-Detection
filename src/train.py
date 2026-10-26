from pathlib import Path
import random
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

SEED = 42
EPOCHS = 3
BATCH_SIZE = 16
NUM_WORKERS = 0  # reliable on Windows
DATA_DIR = Path("data/train_10k")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Device:", device)
print("Device:", device)

if device.type == "cpu":
    print("CPU training selected. Full-dataset training may take many hours.")

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
print("Classes:", dataset.classes)
print("Total images:", len(dataset))

if len(dataset.classes) != 2:
    raise ValueError(
        f"Expected two class folders, found {dataset.classes}"
    )

if set(dataset.classes) != {"fake", "real"}:
    raise ValueError(f"Unexpected class names: {dataset.classes}")

labels = [label for _, label in dataset.samples]
counts = torch.bincount(torch.tensor(labels), minlength=2)
print("Class counts:", dict(zip(dataset.classes, counts.tolist())))

# Random image-level split; for rigorous evaluation, split by source video.
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)
print(f"Training images: {train_size}; validation images: {val_size}")

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda")
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda")
)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# ImageFolder sorts alphabetically: fake=0, real=1.
class_weights = counts.sum().float() / (2 * counts.float().clamp(min=1))
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for step, (images, targets) in enumerate(train_loader, 1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if step % 200 == 0:
            print(
                f"Epoch {epoch+1}/{EPOCHS} "
                f"batch {step}/{len(train_loader)} "
                f"loss={loss.item():.4f}",
                flush=True
            )

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            val_loss += criterion(logits, targets).item()
            correct += (logits.argmax(1) == targets).sum().item()
            total += targets.size(0)

    train_loss = running_loss / max(1, len(train_loader))
    val_loss /= max(1, len(val_loader))
    accuracy = correct / max(1, total)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_loss:.4f} | "
        f"val_accuracy={accuracy:.4f}",
        flush=True
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            "state_dict": model.state_dict(),
            "classes": dataset.classes,
            "epoch": epoch + 1,
            "val_loss": val_loss,
        }, MODEL_DIR / "resnet18_baseline.pth")
        print("Best checkpoint saved.", flush=True)

print("Training complete. Check models/resnet18_baseline.pth")