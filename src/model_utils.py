from pathlib import Path
import torch
from torch import nn
from torchvision.models import resnet18

def build_resnet18():
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

def load_checkpoint(model, checkpoint_path, device):
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict):
        state = checkpoint.get("state_dict", checkpoint.get("model_state_dict",
                          checkpoint.get("model", checkpoint)))
    else:
        state = checkpoint
    if not isinstance(state, dict):
        raise RuntimeError("Unsupported checkpoint format: expected a state dict.")
    state = {k.removeprefix("module."): v for k, v in state.items()
             if isinstance(k, str)}
    try:
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise RuntimeError(
            "This checkpoint does not match a standard ResNet18 with a 2-class head. "
            "It may be from a different architecture or have different layer names. "
            f"Original error: {exc}"
        ) from exc
    model.to(device).eval()
    return model

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
