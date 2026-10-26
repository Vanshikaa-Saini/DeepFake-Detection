from pathlib import Path
import sys
import torch
from torch import nn
from PIL import Image
from torchvision import transforms, models

MODEL_PATH = Path("models/resnet18_smoketest.pth")

if len(sys.argv) < 2:
    print("Usage: python src/predict.py path/to/image.png")
    sys.exit(1)

image_path = Path(sys.argv[1])

if not image_path.is_file():
    raise FileNotFoundError(f"Image not found: {image_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
classes = checkpoint["classes"]

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(checkpoint["state_dict"])
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

image = Image.open(image_path).convert("RGB")
tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    probabilities = torch.softmax(model(tensor), dim=1)[0]
    index = probabilities.argmax().item()

print("Prediction:", classes[index].upper())
print(f"Model score: {probabilities[index].item():.2%}")
print("Note: This smoke-test model is not validated for real-world detection.")