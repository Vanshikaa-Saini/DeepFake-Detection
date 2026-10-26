
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms


MODEL_PATH = Path("models/resnet18_baseline.pth")
IMAGE_SIZE = 160
SAMPLE_EVERY_SECONDS = 1
BATCH_SIZE = 16

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


def main():
    parser = argparse.ArgumentParser(
        description="Predict real/fake from a video using ResNet18."
    )
    parser.add_argument("video", help="Path to the input video")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print("Device:", device)

    checkpoint = torch.load(
        MODEL_PATH, map_location="cpu", weights_only=False
    )
    classes = checkpoint["classes"]
    fake_index = classes.index("fake")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not fps or fps <= 0:
        cap.release()
        raise RuntimeError("Could not determine video frame rate.")

    duration = total_frames / fps
    sample_step = max(1, round(fps * SAMPLE_EVERY_SECONDS))

    print(f"Video duration: {duration:.1f} seconds")
    print(f"Sampling approximately every {SAMPLE_EVERY_SECONDS} second(s)")

    frames = []
    frame_numbers = []
    frame_index = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_index % sample_step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(transform(rgb))
            frame_numbers.append(frame_index)

        frame_index += 1

    cap.release()

    if not frames:
        raise RuntimeError("No frames could be extracted from the video.")

    fake_probabilities = []

    with torch.no_grad():
        for start in range(0, len(frames), BATCH_SIZE):
            batch = torch.stack(
                frames[start:start + BATCH_SIZE]
            ).to(device)

            logits = model(batch)
            probabilities = torch.softmax(logits, dim=1)
            fake_probabilities.extend(
                probabilities[:, fake_index].cpu().tolist()
            )

    mean_fake_probability = float(np.mean(fake_probabilities))
    fake_frame_percentage = (
        100 * sum(p >= 0.5 for p in fake_probabilities)
        / len(fake_probabilities)
    )

    prediction = (
        "FAKE-LIKELY"
        if mean_fake_probability >= 0.5
        else "REAL-LIKELY"
    )

    print("\n--- VIDEO PREDICTION ---")
    print("Video:", video_path.name)
    print("Sampled frames:", len(fake_probabilities))
    print(f"Average fake probability: {mean_fake_probability:.4f}")
    print(f"Frames classified as fake: {fake_frame_percentage:.1f}%")
    print("Prediction:", prediction)
    print("\nNote: This is a frame-aggregation estimate, not a")
    print("verified probability that the entire video is fake.")


if __name__ == "__main__":
    main()
