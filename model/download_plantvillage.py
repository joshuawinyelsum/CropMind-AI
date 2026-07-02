import os
import shutil
import subprocess
from pathlib import Path
import random

DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TRAIN_RATIO = 0.8
MAX_PER_CLASS = 1000
SEED = 42

# Map PlantVillage folder names to CropMind class names
PV_TO_CROPMIND = {
    "Tomato___Bacterial_spot": "Tomato___Bacterial_spot",
    "Tomato___Early_blight": "Tomato___Early_blight",
    "Tomato___Late_blight": "Tomato___Late_blight",
    "Tomato___Leaf_Mold": "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot": "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato___Spider_mites",
    "Tomato___Target_Spot": "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato___Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus": "Tomato___Mosaic_virus",
    "Tomato___healthy": "Tomato___healthy"
}

def main():
    random.seed(SEED)
    print("=" * 60)
    print("PlantVillage Dataset Downloader for CropMind AI")
    print("=" * 60)
    
    repo_url = "https://github.com/spMohanty/PlantVillage-Dataset.git"
    tmp_dir = Path("tmp_plantvillage")
    
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
        
    print("Cloning repository (sparse checkout)...")
    try:
        subprocess.run(["git", "clone", "--filter=blob:none", "--sparse", repo_url, str(tmp_dir)], check=True)
        subprocess.run(["git", "sparse-checkout", "set", "raw/color"], cwd=tmp_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")
        return

    raw_color = tmp_dir / "raw" / "color"
    
    total_train = 0
    total_val = 0

    for pv_name, cropmind_name in PV_TO_CROPMIND.items():
        src_cls = raw_color / pv_name
        if not src_cls.exists():
            print(f"Warning: {pv_name} not found in PlantVillage.")
            continue
            
        images = [f for f in src_cls.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        random.shuffle(images)
        
        # Balance dataset to prevent extreme class imbalance
        images = images[:MAX_PER_CLASS]
        
        split_idx = int(len(images) * TRAIN_RATIO)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]
        
        train_cls_dir = TRAIN_DIR / cropmind_name
        val_cls_dir = VAL_DIR / cropmind_name
        train_cls_dir.mkdir(parents=True, exist_ok=True)
        val_cls_dir.mkdir(parents=True, exist_ok=True)
        
        for img in train_imgs:
            shutil.copy(str(img), str(train_cls_dir / img.name))
        for img in val_imgs:
            shutil.copy(str(img), str(val_cls_dir / img.name))
            
        print(f"  {cropmind_name}: Copied {len(train_imgs)} train, {len(val_imgs)} val.")
        total_train += len(train_imgs)
        total_val += len(val_imgs)
        
    print("Cleaning up...")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("\n" + "=" * 60)
    print("PlantVillage download and balancing complete.")
    print(f"Total Train: {total_train}")
    print(f"Total Val: {total_val}")
    print("=" * 60)

if __name__ == "__main__":
    main()
