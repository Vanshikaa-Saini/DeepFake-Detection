from pathlib import Path
import random
import re

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "dfdc_faces" / "validation"
CHECKPOINT = ROOT / "models" / "cnn_lstm_temporal.pth"
OUTPUT = ROOT / "outputs"
SEQ_LEN = 4
MAX_PER_CLASS = 30
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])
        for p in self.cnn.parameters():
            p.requires_grad = False
        self.lstm = nn.LSTM(512, 128, batch_first=True)
        self.classifier = nn.Linear(128, 2)

    def forward(self, x):
        batch, steps, channels, height, width = x.shape
        x = x.reshape(batch * steps, channels, height, width)
        with torch.no_grad():
            features = self.cnn(x).flatten(1)
        features = features.reshape(batch, steps, 512)
        _, (hidden, _) = self.lstm(features)
        return self.classifier(hidden[-1])


def collect_sequences():
    pattern = re.compile(r"^(.+)_([0-9]+)_([0-9]+)$")
    sequences = []

    for label_id, label in enumerate(["fake", "real"]):
        folder = DATA / label
        groups = {}

        for path in folder.glob("*.png"):
            match = pattern.match(path.stem)
            if not match:
                continue
            video_id, frame_index, face_index = match.groups()
            groups.setdefault(video_id, {}).setdefault(
                int(frame_index), []
            ).append((int(face_index), path))

        candidates = []
        for video_id, frame_groups in groups.items():
            frames = [
                sorted(faces, key=lambda item: item[0])[0][1]
                for _, faces in sorted(frame_groups.items())
            ]
            if len(frames) < SEQ_LEN:
                continue

            positions = [
                round(i * (len(frames) - 1) / (SEQ_LEN - 1))
                for i in range(SEQ_LEN)
            ]
            candidates.append(
                (label_id, video_id, [frames[i] for i in positions])
            )

        random.Random(SEED).shuffle(candidates)
        chosen = candidates[:MAX_PER_CLASS]
        sequences.extend(chosen)
        print(f"{label}: {len(chosen)} sequences selected "
              f"from {len(candidates)} available")

    random.Random(SEED).shuffle(sequences)
    return sequences


def main():
    random.seed(SEED)
    OUTPUT.mkdir(exist_ok=True)

    checkpoint = torch.load(
        CHECKPOINT, map_location=DEVICE, weights_only=True
    )
    model = CNNLSTM().to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    sequences = collect_sequences()
    if not sequences:
        raise RuntimeError("No valid sequences found in validation data.")

    y_true, y_pred = [], []
    class_names = checkpoint.get("classes", ["fake", "real"])

    with torch.no_grad():
        for label_id, video_id, paths in sequences:
            frames = []
            for path in paths:
                with Image.open(path) as image:
                    frames.append(transform(image.convert("RGB")))

            x = torch.stack(frames).unsqueeze(0).to(DEVICE)
            probs = torch.softmax(model(x), dim=1)[0]
            prediction = int(probs.argmax().item())

            y_true.append(label_id)
            y_pred.append(prediction)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print("\n=== TEMPORAL MODEL VALIDATION ===")
    print(f"Sequences evaluated: {len(y_true)}")
    print(f"Accuracy:  {accuracy:.4f}")
    print("\nPer-class metrics:")
    print(classification_report(
        y_true, y_pred,
        labels=[0, 1],
        target_names=class_names,
        zero_division=0,
    ))
    print("Confusion matrix (rows=true, columns=predicted):")
    print("Classes:", class_names)
    print(matrix)

    report = (
        f"Sequences evaluated: {len(y_true)}\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Precision fake: {precision[0]:.4f}\n"
        f"Recall fake: {recall[0]:.4f}\n"
        f"F1 fake: {f1[0]:.4f}\n"
        f"Precision real: {precision[1]:.4f}\n"
        f"Recall real: {recall[1]:.4f}\n"
        f"F1 real: {f1[1]:.4f}\n"
        f"Confusion matrix:\n{matrix}\n"
    )
    report_path = OUTPUT / "temporal_evaluation.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nMetrics saved to: {report_path}")


if __name__ == "__main__":
    main()