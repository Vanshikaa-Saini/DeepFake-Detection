import argparse
from PIL import Image
import torch
from torchvision import transforms
from .model_utils import build_resnet18, load_checkpoint, get_device

def main():
    parser = argparse.ArgumentParser(description="Predict REAL/FAKE for one image with a saved ResNet18.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    device = get_device()
    model = load_checkpoint(build_resnet18(), args.checkpoint, device)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    with Image.open(args.image) as im:
        x = transform(im.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    # Training convention used by the project: class index 0=REAL, 1=FAKE.
    idx = int(probs.argmax().item())
    labels = ["REAL", "FAKE"]
    print(f"Prediction: {labels[idx]}")
    print(f"REAL score: {probs[0].item():.4f}")
    print(f"FAKE score: {probs[1].item():.4f}")
if __name__ == "__main__":
    main()
