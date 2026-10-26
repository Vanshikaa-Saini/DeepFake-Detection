from pathlib import Path
import numpy as np
import cv2
import mediapipe as mp

def extract_landmarks(image_path):
    """Return normalized face-landmark coordinates for one image."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # MediaPipe Face Mesh landmark detector
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
    ) as face_mesh:
        result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None

    points = result.multi_face_landmarks[0].landmark
    coords = np.array([[p.x, p.y, p.z] for p in points], dtype=np.float32)

    # Center and scale coordinates to reduce dependence on face position.
    coords[:, :2] -= coords[:, :2].mean(axis=0, keepdims=True)
    scale = np.std(coords[:, :2])
    if scale > 1e-6:
        coords[:, :2] /= scale

    return coords

if __name__ == "__main__":
    image_path = input("Enter image path: ").strip().strip('"')
    points = extract_landmarks(Path(image_path))
    if points is None:
        print("No face detected.")
    else:
        print("Landmark array shape:", points.shape)
        print("Landmarks extracted successfully.")