# dataset.py
# Clean, standard dataset loader using torchvision.datasets.ImageFolder.
# Single source of truth is delegated entirely to the folder structure and model checkpoint.
#
# Strong augmentation pipeline for agricultural disease classification:
#   - RandomResizedCrop (scale variation, simulates distance)
#   - RandomHorizontalFlip + RandomVerticalFlip (leaf orientation invariance)
#   - RandomRotation (field camera angles)
#   - ColorJitter (lighting, weather, camera sensor variation)
#   - GaussianBlur (simulates motion blur, low-res cameras)
#   - RandomAffine (perspective distortion)

from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_transform, val_transform


def build_dataloaders(
    data_dir: str = "data",
    batch_size: int = 16,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, list[str]]:
    dataset_root = Path(data_dir)
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"

    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    train_transform, val_transform = build_transforms()

    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=str(val_dir), transform=val_transform)

    generator = torch.Generator()
    generator.manual_seed(seed)

    targets = train_dataset.targets
    class_counts = torch.bincount(torch.tensor(targets))
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[targets]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,  # CRITICAL: keep num_workers=0 on Windows
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # CRITICAL: keep num_workers=0 on Windows
        pin_memory=False,
    )

    return train_loader, val_loader, train_dataset.classes
