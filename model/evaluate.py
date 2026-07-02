import torch
import torch.nn as nn
from dataset import build_dataloaders
from model import build_resnet34
from train import validate

def main():
    device = torch.device("cpu")
    print("Loading data with batch_size=1 to avoid OOM...")
    train_loader, val_loader, class_names = build_dataloaders(batch_size=1, num_workers=0)
    
    print("Loading model...")
    model = build_resnet34(num_classes=len(class_names), pretrained=False, freeze_backbone=True).to(device)
    
    checkpoint_path = "models/tomato_model_finetuned.pth"
    print(f"Loading checkpoint {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    print(f"Model loaded. Checkpoint was from Epoch {checkpoint.get('epoch', 'unknown')} with Val Acc: {checkpoint.get('val_accuracy', 'unknown')}")
    
    criterion = nn.CrossEntropyLoss()
    print("Running validation...")
    validate(model, val_loader, criterion, device, class_names)

if __name__ == "__main__":
    main()
