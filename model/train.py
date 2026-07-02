# train.py
# Clean training script using checkpoint-driven single source of truth class mapping.

import argparse
import os
from pathlib import Path
from collections import Counter

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim

torch.set_num_threads(2)
torch.backends.mkldnn.enabled = False
import gc

from dataset import build_dataloaders
from model import build_resnet34

# ── Locked production hyperparameters ─────────────────────────────────────────
SEED: int = 42
EPOCHS: int = 3
BATCH_SIZE: int = 2
ACCUMULATION_STEPS: int = 4
LEARNING_RATE: float = 3e-5
PATIENCE: int = 5
LR_PATIENCE: int = 2
LR_FACTOR: float = 0.3
# ──────────────────────────────────────────────────────────────────────────────


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    total_samples = 0
    
    optimizer.zero_grad(set_to_none=True)

    for step, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Scale loss by accumulation steps
        scaled_loss = loss / ACCUMULATION_STEPS
        scaled_loss.backward()
        
        if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        # Force deallocation of references
        del outputs, loss, scaled_loss, images, labels
        gc.collect()

    return running_loss / max(total_samples, 1)


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
) -> tuple[float, float]:
    model.eval()
    correct = 0
    top3_correct = 0
    total = 0
    running_loss = 0.0
    num_classes = len(class_names)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            
            _, top3 = outputs.topk(3, 1, True, True)
            top3_correct += (top3 == labels.view(-1, 1).expand_as(top3)).sum().item()
            
            total += labels.size(0)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion[t.long(), p.long()] += 1

            del outputs, loss, preds, top3, images, labels
            gc.collect()

    val_loss = running_loss / max(total, 1)
    val_acc = correct / max(total, 1)
    top3_acc = top3_correct / max(total, 1)

    print("\nConfusion Matrix (Rows=True, Cols=Predicted):")
    print(confusion)
    print("\nPer-class Accuracy:")
    for i, name in enumerate(class_names):
        row_total = confusion[i].sum().item()
        row_correct = confusion[i, i].item()
        if row_total > 0:
            print(f"  {name}: {row_correct}/{row_total} ({100.0 * row_correct / row_total:.1f}%)")
        else:
            print(f"  {name}: No samples in val set")
    print(f"\nTop-3 Accuracy: {top3_acc:.4f}\n")

    return val_loss, val_acc


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


def validate_checkpoint_artifact(checkpoint: dict) -> bool:
    required_keys = ["model_state_dict", "class_names", "class_to_idx"]

    for key in required_keys:
        if key not in checkpoint:
            raise ValueError(f"Checkpoint missing required field: {key}")

    class_names = checkpoint["class_names"]
    class_to_idx = checkpoint["class_to_idx"]

    if not isinstance(class_names, list) or not all(isinstance(name, str) for name in class_names):
        raise ValueError("Checkpoint class_names must be a list of strings")
    if not isinstance(class_to_idx, dict):
        raise ValueError("Checkpoint class_to_idx must be a dict")

    if len(set(class_to_idx.keys())) != len(class_to_idx):
        raise ValueError("Duplicate class mapping detected")

    if len(set(class_names)) != len(class_names):
        raise ValueError("Duplicate class names detected")

    for class_name, index in class_to_idx.items():
        if not isinstance(class_name, str):
            raise ValueError("Checkpoint class_to_idx keys must be strings")
        if not isinstance(index, int):
            raise ValueError("Checkpoint class_to_idx values must be integers")
        if class_name not in class_names:
            raise ValueError(f"Checkpoint class_to_idx contains unknown class: {class_name}")

    assert_strict_bijection(class_names, class_to_idx)

    return True


def save_checkpoint(
    model: nn.Module,
    class_names: list[str],
    class_to_idx: dict[str, int],
    output_path: str,
    epoch: int,
    val_accuracy: float,
) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save training metadata directly in the checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "num_classes": len(class_names),
        "epoch": epoch,
        "val_accuracy": val_accuracy,
        "seed": SEED,
    }
    validate_checkpoint_artifact(checkpoint)
    torch.save(checkpoint, output_file)

    saved_checkpoint = torch.load(output_file, map_location="cpu", weights_only=False)
    validate_checkpoint_artifact(saved_checkpoint)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CropMind AI tomato disease classifier.")
    parser.add_argument("--data-dir", default="data", help="Root directory with train/ and val/.")
    parser.add_argument("--output", default="models/tomato_model_finetuned.pth", help="Checkpoint save path.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze ResNet18 backbone (train head only). Good for CPU environments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    _set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Seed: {SEED} | Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LEARNING_RATE}")

    train_loader, val_loader, class_names = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=BATCH_SIZE,
        num_workers=args.num_workers,
        seed=SEED,
    )

    train_dataset = train_loader.dataset
    class_names = list(train_dataset.classes)
    class_to_idx = dict(train_dataset.class_to_idx)
    print("\n--- Training class_to_idx from train_dataset ---")
    for name, idx in class_to_idx.items():
        print(f"  {name} -> {idx}")
    print("--------------------------------------\n")

    assert_strict_bijection(class_names, class_to_idx)

    # Model
    model = build_resnet34(
        num_classes=len(class_names),
        pretrained=True,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    if os.path.exists("models/tomato_model_finetuned.pth"):
        checkpoint = torch.load("models/tomato_model_finetuned.pth", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Resumed from models/tomato_model_finetuned.pth")
    elif os.path.exists("models/tomato_model.pth"):
        checkpoint = torch.load("models/tomato_model.pth", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Resumed from models/tomato_model.pth")

    targets = train_dataset.targets
    class_counts = Counter(targets)
    total = sum(class_counts.values())
    weights = [total / class_counts[i] for i in range(len(class_counts))]
    weights = torch.tensor(weights, dtype=torch.float).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE
    )

    best_val_accuracy = 0.0
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_accuracy = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            class_names=class_names,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f}")
        
        if train_loss < 0.3 and val_accuracy < 0.6:
            print("WARNING: Overfitting detected")

        scheduler.step(val_loss)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            save_checkpoint(
                model=model,
                class_names=class_names,
                class_to_idx=class_to_idx,
                output_path=args.output,
                epoch=epoch,
                val_accuracy=val_accuracy,
            )
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/best_model.pth")
            print(f"  [SUCCESS] Saved best model to {args.output}\n")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement. Early-stop patience: {epochs_without_improvement}/{PATIENCE}\n")
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    print(f"\nTraining complete. Best val accuracy: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    main()
