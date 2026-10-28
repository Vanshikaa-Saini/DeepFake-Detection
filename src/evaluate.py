
import argparse
import json
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from .model_utils import build_resnet18, load_checkpoint, get_device


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved ResNet18 deepfake detector."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (data_dir / "real").is_dir() or not (data_dir / "fake").is_dir():
        raise FileNotFoundError(
            f"Expected both folders: {data_dir / 'real'} and {data_dir / 'fake'}"
        )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = datasets.ImageFolder(str(data_dir), transform=transform)

    if set(dataset.class_to_idx) != {"real", "fake"}:
        raise RuntimeError(
            f"Expected real/ and fake/ folders; found {dataset.class_to_idx}"
        )

    # Convert dataset labels to the project's intended convention:
    # real=0, fake=1.
    original_mapping = dataset.class_to_idx
    remap = {
        original_mapping["real"]: 0,
        original_mapping["fake"]: 1,
    }

    dataset.samples = [
        (path, remap[label]) for path, label in dataset.samples
    ]
    dataset.targets = [label for _, label in dataset.samples]
    dataset.class_to_idx = {"real": 0, "fake": 1}

    if len(dataset) == 0:
        raise RuntimeError(f"No images found in {data_dir}")

    print(f"Images found: {len(dataset)}")
    print(f"Original folder mapping: {original_mapping}")
    print(f"Evaluation mapping: {dataset.class_to_idx}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = get_device()
    print(f"Device: {device}")
    print("Loading saved checkpoint...", flush=True)

    model = load_checkpoint(
        build_resnet18(),
        args.checkpoint,
        device,
    )
    model.eval()

    ys, preds, fake_scores = [], [], []

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            if batch_idx % 50 == 0:
                print(
                    f"Evaluating batch {batch_idx}/{len(loader)}",
                    flush=True,
                )

            x = x.to(device)
            logits = model(x)
            probabilities = torch.softmax(logits, dim=1)

            preds.extend(logits.argmax(dim=1).cpu().tolist())
            ys.extend(y.tolist())
            fake_scores.extend(probabilities[:, 1].cpu().tolist())

    if not ys or not preds:
        raise RuntimeError(
            "No predictions were collected. Check the dataset and inference loop."
        )

    print(f"Collected predictions for {len(ys)} images.", flush=True)

    accuracy = accuracy_score(ys, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        ys,
        preds,
        average="binary",
        zero_division=0,
    )
    cm = confusion_matrix(ys, preds, labels=[0, 1])

    report = {
        "num_images": len(dataset),
        "class_to_idx": {"real": 0, "fake": 1},
        "accuracy": float(accuracy),
        "fake_precision": float(precision),
        "fake_recall": float(recall),
        "fake_f1": float(f1),
        "confusion_matrix_labels": ["real", "fake"],
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            ys,
            preds,
            labels=[0, 1],
            target_names=["real", "fake"],
            zero_division=0,
        ),
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Real", "Fake"])
    ax.set_yticks([0, 1], ["Real", "Fake"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title("ResNet18 Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
            )

    fig.tight_layout()
    confusion_path = output_dir / "confusion_matrix.png"
    fig.savefig(confusion_path, dpi=160)
    plt.close(fig)

    print("\n===== EVALUATION RESULTS =====")
    print(f"Images evaluated: {len(ys)}")
    print(f"Accuracy:         {accuracy:.4f}")
    print(f"Fake precision:   {precision:.4f}")
    print(f"Fake recall:      {recall:.4f}")
    print(f"Fake F1 score:    {f1:.4f}")
    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print("                 Predicted Real  Predicted Fake")
    print(f"Actual Real      {cm[0, 0]:14d}  {cm[0, 1]:14d}")
    print(f"Actual Fake      {cm[1, 0]:14d}  {cm[1, 1]:14d}")
    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Confusion matrix saved to: {confusion_path}")


if __name__ == "__main__":
    main()
