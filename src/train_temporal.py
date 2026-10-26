from pathlib import Path
import random
import re

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms, models
from PIL import Image

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

ROOT = Path("data/raw/dfdc_faces/train")
OUT = Path("models")
OUT.mkdir(exist_ok=True)

SEQ_LEN = 4
MAX_PER_CLASS = 30
BATCH_SIZE = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


def parse_name(path):
    # Expected filename: videoID_frameIndex_faceIndex.png
    match = re.match(r"^(.+)_([0-9]+)_([0-9]+)$", path.stem)
    if not match:
        return None
    return match.group(1), int(match.group(2)), int(match.group(3))


def collect_sequences():
    result = []
    for label, class_name in enumerate(["fake", "real"]):
        folder = ROOT / class_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing class folder: {folder}")

        videos = {}
        for path in folder.glob("*.png"):
            parsed = parse_name(path)
            if parsed is None:
                continue
            video_id, frame_idx, face_idx = parsed
            videos.setdefault(video_id, {}).setdefault(frame_idx, []).append(
                (face_idx, path)
            )

        candidates = []
        for video_id, frames in videos.items():
            ordered = [
                sorted(faces, key=lambda item: item[0])[0][1]
                for _, faces in sorted(frames.items())
            ]
            if len(ordered) >= SEQ_LEN:
                # Sample frames across the available sequence.
                positions = [
                    round(i * (len(ordered) - 1) / (SEQ_LEN - 1))
                    for i in range(SEQ_LEN)
                ]
                candidates.append(([ordered[i] for i in positions], label))

        random.shuffle(candidates)
        result.extend(candidates[:MAX_PER_CLASS])
        print(f"{class_name}: {len(candidates)} usable video groups; "
              f"selected {min(len(candidates), MAX_PER_CLASS)}")

    random.shuffle(result)
    return result


class FaceSequences(Dataset):
    def __init__(self, items):
        self.items = items
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        paths, label = self.items[index]
        frames = []
        for path in paths:
            with Image.open(path) as image:
                frames.append(self.transform(image.convert("RGB")))
        return torch.stack(frames), label


items = collect_sequences()
if len(items) < 4:
    raise RuntimeError(
        "Too few valid sequences. Check the filename pattern and dataset."
    )

dataset = FaceSequences(items)
train_n = max(1, int(0.8 * len(dataset)))
val_n = len(dataset) - train_n
train_set, val_set = random_split(
    dataset, [train_n, val_n],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)


class CNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])
        for parameter in self.cnn.parameters():
            parameter.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=512, hidden_size=128,
            batch_first=True
        )
        self.classifier = nn.Linear(128, 2)

    def forward(self, x):
        batch, steps, channels, height, width = x.shape
        x = x.reshape(batch * steps, channels, height, width)

        # Frozen CNN extracts features; no gradients needed here.
        with torch.no_grad():
            features = self.cnn(x).flatten(1)

        features = features.reshape(batch, steps, 512)
        _, (hidden, _) = self.lstm(features)
        return self.classifier(hidden[-1])


model = CNNLSTM().to(device)
optimizer = torch.optim.Adam(
    list(model.lstm.parameters()) +
    list(model.classifier.parameters()),
    lr=0.001
)
criterion = nn.CrossEntropyLoss()

for epoch in range(2):
    model.train()
    total_loss = 0.0

    for sequences, labels in train_loader:
        sequences = sequences.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(sequences)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/2 | "
        f"training loss: {total_loss / max(1, len(train_loader)):.4f}"
    )

model.eval()
correct, total = 0, 0
with torch.no_grad():
    for sequences, labels in val_loader:
        logits = model(sequences.to(device))
        predictions = logits.argmax(dim=1).cpu()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

print(f"Prototype validation accuracy: {correct / max(total, 1):.4f}")

torch.save({
    "state_dict": model.state_dict(),
    "classes": ["fake", "real"],
    "sequence_length": SEQ_LEN,
}, OUT / "cnn_lstm_temporal.pth")

print("Saved models/cnn_lstm_temporal.pth")