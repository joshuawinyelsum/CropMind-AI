# inference.py
# Standalone CLI tool for single-image prediction on CPU.
# Uses the model class mappings loaded completely from the checkpoint (single source of truth).

import argparse
import json
import os
from pathlib import Path
import io

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
from PIL import Image

torch.set_num_threads(1)

from torchvision import transforms
from model.model import build_resnet34

# OOD threshold (must match pytorch_model.py)
CONFIDENCE_THRESHOLD = 0.60
ENTROPY_THRESHOLD = 0.50


def assert_strict_bijection(class_names: list[str], class_to_idx: dict[str, int]) -> dict[int, str]:
    if len(class_names) != len(class_to_idx):
        raise ValueError("Class list and index map size mismatch")

    expected_indices = set(range(len(class_names)))
    actual_indices = set(class_to_idx.values())

    if expected_indices != actual_indices:
        raise ValueError(
            f"Non-contiguous or invalid indices detected: {actual_indices}"
        )

    idx_to_class = {index: class_name for class_name, index in class_to_idx.items()}

    for index, class_name in enumerate(class_names):
        mapped_class = idx_to_class.get(index)
        if mapped_class != class_name:
            raise ValueError(
                f"Broken mapping at index {index}: {class_name} != {mapped_class}"
            )

    return idx_to_class


def assert_checkpoint_integrity(class_names: list[str], class_to_idx: dict[str, int]) -> None:
    if not isinstance(class_names, list) or not all(isinstance(name, str) for name in class_names):
        raise ValueError("Checkpoint 'class_names' must be a list of strings.")
    if not isinstance(class_to_idx, dict):
        raise ValueError("Checkpoint 'class_to_idx' must be a dict.")
    if len(class_names) != len(class_to_idx):
        raise ValueError("Checkpoint class_names and class_to_idx length mismatch.")
    if len(set(class_names)) != len(class_names):
        raise ValueError("Checkpoint class_names contains duplicates.")

    indices = list(class_to_idx.values())
    if not all(isinstance(index, int) for index in indices):
        raise ValueError("Checkpoint class_to_idx values must be integers.")
    if len(set(indices)) != len(indices):
        raise ValueError("Checkpoint class_to_idx is not one-to-one.")
    for class_name in class_to_idx.keys():
        if not isinstance(class_name, str):
            raise ValueError("Checkpoint class_to_idx keys must be strings.")
        if class_name not in class_names:
            raise ValueError(f"Checkpoint class_to_idx contains unknown class: {class_name}")

    assert_strict_bijection(class_names, class_to_idx)


def _compute_entropy(probabilities: torch.Tensor) -> float:
    eps = 1e-9
    return float(-torch.sum(probabilities * torch.log(probabilities + eps)).item())


def load_model(model_path: str) -> tuple[nn.Module, list[str], dict[str, int]]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    class_names = checkpoint.get("class_names")
    class_to_idx = checkpoint.get("class_to_idx")
    state_dict = checkpoint.get("model_state_dict")

    if not class_names:
        raise ValueError("Checkpoint missing class_names.")
    if not class_to_idx:
        raise ValueError("Checkpoint missing class_to_idx.")
    if state_dict is None:
        raise ValueError("Checkpoint missing model_state_dict.")

    assert_checkpoint_integrity(class_names, class_to_idx)

    model = build_resnet34(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(state_dict)
    model.eval()

    return model, class_names, class_to_idx


def predict(image_bytes: bytes, model_path: str) -> dict:
    model, class_names, class_to_idx = load_model(model_path)
    idx_to_class = assert_strict_bijection(class_names, class_to_idx)

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = val_transform(image).unsqueeze(0)  # (1, 3, 224, 224)

    with torch.no_grad():
        logits = model(input_tensor)
        print("OUTPUT SHAPE:", logits.shape)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)

    confidence_val = round(float(confidence.item()), 4)
    entropy_val = _compute_entropy(probabilities.squeeze(0))

    pred_idx = int(predicted_idx.item())
    class_name = idx_to_class[pred_idx]

    print("RAW PRED INDEX:", pred_idx)
    print("CLASS NAME:", class_name)

    # OOD check (TEMPORARILY DISABLED)
    # is_ood = (confidence_val < CONFIDENCE_THRESHOLD) or (entropy_val > ENTROPY_THRESHOLD)
    # 
    # if is_ood:
    #     return {
    #         "is_tomato": False,
    #         "predicted_class": "Unknown crop",
    #         "confidence": confidence_val,
    #         "disease_key": "unsupported_crop",
    #         "treatments": [],
    #         "explanation_map": None,
    #     }

    return {
        "predicted_class": class_name,
        "confidence": float(confidence_val)
    }


def main():
    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument("image_path", help="Path to the image to classify.")
    parser.add_argument(
        "--model-path",
        default="models/tomato_model.pth",
        help="Path to the trained model checkpoint.",
    )
    args = parser.parse_args()

    try:
        with open(args.image_path, "rb") as f:
            image_bytes = f.read()
        result = predict(image_bytes, args.model_path)
        print(json.dumps(result, indent=2))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))


def predict_image(image_bytes: bytes):
    print("DEBUG -> predict_image CALLED")
    res = predict(image_bytes, "models/tomato_model.pth")
    
    print("DEBUG -> class_name:", res["predicted_class"])
    print("DEBUG -> confidence:", res["confidence"])
    
    return {
        "predicted_class": res["predicted_class"],
        "confidence": res["confidence"]
    }

if __name__ == "__main__":
    import sys
    with open(sys.argv[1], "rb") as f:
        print(predict_image(f.read()))
