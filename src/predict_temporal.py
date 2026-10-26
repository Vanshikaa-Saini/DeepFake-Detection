from pathlib import Path
import random
import re

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "dfdc_faces" / "validation"
CHECKPOINT = ROOT / "models" / "cnn_lstm_temporal.pth"
SEQ_LEN = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])

        for parameter in self.cnn.parameters():
            parameter.requires_grad = False

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


def find_sequences():
    pattern = re.compile(r"^(.+)_([0-9]+)_([0-9]+)$")
    sequences = []

    for label in ("fake", "real"):
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

        for video_id, frame_groups in groups.items():
            frames = [
                sorted(faces, key=lambda item: item[0])[0][1]
                for _, faces in sorted(frame_groups.items())
            ]

            if len(frames) >= SEQ_LEN:
                positions = [
                    round(i * (len(frames) - 1) / (SEQ_LEN - 1))
                    for i in range(SEQ_LEN)
                ]
                selected = [frames[i] for i in positions]
                sequences.append((label, video_id, selected))

    return sequences


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {CHECKPOINT}")

    checkpoint = torch.load(
        CHECKPOINT, map_location=DEVICE, weights_only=True
    )

    model = CNNLSTM().to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    sequences = find_sequences()
    if not sequences:
        print("No sequences with at least four frames were found.")
        return

    random.seed(42)
    label, video_id, paths = random.choice(sequences)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    frames = []
    for path in paths:
        with Image.open(path) as image:
            frames.append(transform(image.convert("RGB")))

    x = torch.stack(frames).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probabilities = torch.softmax(model(x), dim=1)[0].cpu()

    classes = checkpoint.get("classes", ["fake", "real"])
    prediction = classes[int(probabilities.argmax())]

    print("\n--- Deepfake Detection Prototype ---")
    print("Video group:", video_id)
    print("Reference dataset label:", label)
    print("Frames:", [p.name for p in paths])
    print("Predicted label:", prediction.upper())
    print(f"Fake score: {probabilities[0].item():.4f}")
    print(f"Real score: {probabilities[1].item():.4f}")
    print("\nNote: Experimental prediction, not a validated forensic result.")


if __name__ == "__main__":
    main()