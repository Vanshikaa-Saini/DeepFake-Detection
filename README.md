# Offline Deepfake Detection (existing checkpoint; no training)

This package is for continuing offline with the files you already have:
- `resnet18_baseline.pth` (or either renamed duplicate)
- your existing image/frame dataset
- optionally `cnn_lstm_temporal.pth` and existing evaluation artifacts

**No `train.py` is needed for the workflow in this package.** It runs inference and evaluates a compatible saved 2D ResNet18 checkpoint. It does not download or require any additional dataset.

## Important distinction
This uses your saved **2D ResNet18 image classifier**. It is not a full reproduction of a paper method that requires separately trained 3D video architectures. Reusing this checkpoint avoids training, but only implements the image-model path. The old CNN-LSTM checkpoint is not used by these scripts.

## 1. Folder layout

Extract this ZIP to a folder, then arrange files like this:

```
deepfake_offline/
├── src/
│   ├── __init__.py
│   ├── model_utils.py
│   ├── predict.py
│   └── evaluate.py
├── models/
│   └── resnet18_baseline.pth
├── data/
│   └── validation/
│       ├── real/
│       │   ├── image1.jpg
│       │   └── ...
│       └── fake/
│           ├── image1.jpg
│           └── ...
├── outputs/
├── requirements.txt
└── README.md
```

Copy your saved checkpoint into `models/`. You can keep the original filename; pass it with `--checkpoint`.
Use your existing extracted face/frame images. No new dataset is downloaded. Evaluation expects `real/` and `fake/` class folders.

## 2. Install dependencies (offline)
If Python packages are already installed, skip this. Otherwise install `torch`, `torchvision`, `pillow`, `scikit-learn`, and `matplotlib` from your local wheel cache or an internet-connected setup before going offline. The scripts themselves make no network requests.

## 3. Predict one image

From the project folder:

```
python -m src.predict --checkpoint models/resnet18_baseline.pth --image path/to/image.jpg
```

## 4. Evaluate your existing validation images

```
python -m src.evaluate --checkpoint models/resnet18_baseline.pth --data-dir data/validation --output-dir outputs
```

The evaluation script writes `metrics.json` and `confusion_matrix.png`. It reports accuracy, precision, recall, F1 and the confusion matrix.

## Checkpoint compatibility
The loader supports common checkpoint formats (`state_dict`, `model_state_dict`, or a raw state dict) and removes a leading `module.` prefix. If your checkpoint was saved from a different architecture or uses different classifier dimensions, it will stop with a clear error instead of pretending the weights loaded.

## Honest limitations
- This is inference/evaluation, not training.
- The model sees individual images/frames, not the full temporal dynamics of a video.
- A score is a model confidence, not proof that media is fake.
- For paper-exact 3D video models, compatible trained checkpoints are required. A 2D ResNet checkpoint cannot be converted into trained 3D weights without additional training.
