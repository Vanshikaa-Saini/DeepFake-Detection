from pathlib import Path
import random

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "dfdc_faces" / "validation"
OUTPUT = ROOT / "outputs" / "landmark_motion_features.npz"
SAMPLES_PER_CLASS = 2
random.seed(42)


def extract_sequence(image_paths):
    frames = []

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
    ) as mesh:
        for path in image_paths:
            image = cv2.imread(str(path))
            if image is None:
                print(f"Skipping unreadable image: {path.name}")
                continue

            result = mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not result.multi_face_landmarks:
                print(f"No face detected: {path.name}")
                continue

            points = np.array(
                [[p.x, p.y, p.z] for p in result.multi_face_landmarks[0].landmark],
                dtype=np.float32,
            )
            points[:, :2] -= points[:, :2].mean(axis=0)
            scale = np.max(np.linalg.norm(points[:, :2], axis=1))
            if scale > 1e-8:
                points[:, :2] /= scale
            points[:, 2] -= points[:, 2].mean()
            frames.append((path, points))

    if not frames:
        return None, []

    coordinates = np.stack([points for _, points in frames])
    motion = np.zeros_like(coordinates)
    if len(coordinates) > 1:
        motion[1:] = coordinates[1:] - coordinates[:-1]

    return np.concatenate([coordinates, motion], axis=-1), [
        str(path) for path, _ in frames
    ]


def main():
    selected = []
    labels = []

    for label in ("fake", "real"):
        folder = DATA / label
        images = sorted(folder.glob("*.png"))

        if len(images) < SAMPLES_PER_CLASS:
            raise FileNotFoundError(f"Not enough images in {folder}")

        chosen = random.sample(images, SAMPLES_PER_CLASS)
        selected.extend(chosen)
        labels.extend([label] * len(chosen))

    print(f"Selected {SAMPLES_PER_CLASS} fake + {SAMPLES_PER_CLASS} real images.")

    all_features, all_paths, all_labels = [], [], []
    for label in ("fake", "real"):
        class_paths = [
            p for p, y in zip(selected, labels) if y == label
        ]
        features, valid_paths = extract_sequence(class_paths)

        if features is not None:
            all_features.append(features)
            all_paths.extend(valid_paths)
            all_labels.extend([label] * len(valid_paths))

    if not all_features:
        print("No faces detected in the selected samples.")
        return

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT,
        features=np.concatenate(all_features, axis=0),
        image_paths=np.array(all_paths),
        labels=np.array(all_labels),
    )
    print("Feature shape:", np.concatenate(all_features, axis=0).shape)
    print("Labels:", all_labels)
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()