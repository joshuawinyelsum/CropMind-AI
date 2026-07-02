# download_plantdoc.py
# Downloads tomato disease images from the PlantDoc GitHub repository
# and organizes them into data/train and data/val with an 80/20 split.
#
# Usage: python download_plantdoc.py
#
# This script will:
# 1. Use the GitHub API to list all images in each PlantDoc tomato folder
# 2. Download raw image files via raw.githubusercontent.com
# 3. Map PlantDoc folder names to CropMind class names
# 4. Split images 80/20 into train/val
# 5. Merge with existing images (skip duplicates by filename)

import os
import sys
import json
import time
import random
import hashlib
import shutil
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

# ── Configuration ──────────────────────────────────────────────────────────────
REPO = "pratikkayal/PlantDoc-Dataset"
BRANCH = "master"
API_BASE = f"https://api.github.com/repos/{REPO}/contents"
RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}"

DATA_DIR = Path("data")
TEST_DIR = DATA_DIR / "test"

# PlantDoc folder name -> CropMind class name
PLANTDOC_TO_CROPMIND = {
    "Tomato leaf bacterial spot":            "Tomato___Bacterial_spot",
    "Tomato Early blight leaf":              "Tomato___Early_blight",
    "Tomato leaf late blight":               "Tomato___Late_blight",
    "Tomato mold leaf":                      "Tomato___Leaf_Mold",
    "Tomato Septoria leaf spot":             "Tomato___Septoria_leaf_spot",
    "Tomato two spotted spider mites leaf":  "Tomato___Spider_mites",
    "Tomato leaf yellow virus":              "Tomato___Yellow_Leaf_Curl_Virus",
    "Tomato leaf mosaic virus":              "Tomato___Mosaic_virus",
    "Tomato leaf":                           "Tomato___healthy",
}

# Classes that PlantDoc does NOT cover (keep existing images)
MISSING_CLASSES = ["Tomato___Target_Spot"]

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}
# ──────────────────────────────────────────────────────────────────────────────


def github_api_get(url, retries=3, delay=2):
    """Fetch JSON from the GitHub API with retries and rate limit handling."""
    headers = {"Accept": "application/vnd.github.v3+json"}
    for attempt in range(retries):
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            if e.code == 403:
                # Rate limited - wait and retry
                print(f"  Rate limited. Waiting {delay * (attempt + 1)}s...")
                time.sleep(delay * (attempt + 1))
            elif e.code == 404:
                print(f"  404 Not Found: {url}")
                return []
            else:
                print(f"  HTTP {e.code} for {url}")
                time.sleep(delay)
        except URLError as e:
            print(f"  Network error: {e}. Retrying...")
            time.sleep(delay)
    print(f"  FAILED after {retries} attempts: {url}")
    return []


def download_file(url, dest_path, retries=3, delay=1):
    """Download a single file with retries."""
    for attempt in range(retries):
        try:
            req = Request(url)
            with urlopen(req, timeout=30) as resp:
                with open(dest_path, "wb") as f:
                    f.write(resp.read())
            return True
        except (HTTPError, URLError, OSError) as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                print(f"    FAILED to download: {url} ({e})")
                return False
    return False


def list_images_in_folder(plantdoc_folder):
    """List all image files in a PlantDoc folder via GitHub API."""
    # URL-encode spaces in folder name
    encoded = plantdoc_folder.replace(" ", "%20")
    
    # Try train folder first
    url = f"{API_BASE}/train/{encoded}"
    items = github_api_get(url)
    
    images = []
    if isinstance(items, list):
        for item in items:
            if item.get("type") == "file":
                name = item["name"]
                ext = os.path.splitext(name)[1]
                if ext in VALID_EXTENSIONS:
                    images.append({
                        "name": name,
                        "download_url": item.get("download_url"),
                        "source": "train",
                    })
    
    # Also check test folder
    url_test = f"{API_BASE}/test/{encoded}"
    items_test = github_api_get(url_test)
    
    if isinstance(items_test, list):
        for item in items_test:
            if item.get("type") == "file":
                name = item["name"]
                ext = os.path.splitext(name)[1]
                if ext in VALID_EXTENSIONS:
                    images.append({
                        "name": name,
                        "download_url": item.get("download_url"),
                        "source": "test",
                    })
    
    return images


def get_existing_filenames(class_dir):
    """Get set of existing filenames in a directory."""
    if not class_dir.exists():
        return set()
    return {f.name for f in class_dir.iterdir() if f.is_file()}


def main():
    print("=" * 60)
    print("PlantDoc Dataset Downloader for CropMind AI")
    print("=" * 60)
    print(f"Source: github.com/{REPO}")
    print(f"Target: {TEST_DIR.resolve()}")
    print(f"Split:  100% test")
    print()
    
    total_downloaded = 0
    total_skipped = 0
    
    for plantdoc_name, cropmind_name in PLANTDOC_TO_CROPMIND.items():
        print(f"\n--- {plantdoc_name} -> {cropmind_name} ---")
        
        # List all images from GitHub
        print(f"  Fetching image list from GitHub API...")
        images = list_images_in_folder(plantdoc_name)
        print(f"  Found {len(images)} images in PlantDoc")
        
        if not images:
            print(f"  WARNING: No images found. Skipping.")
            continue
        
        # Get existing filenames to avoid duplicates
        test_class_dir = TEST_DIR / cropmind_name
        test_class_dir.mkdir(parents=True, exist_ok=True)
        
        existing_test = get_existing_filenames(test_class_dir)
        
        # Filter out already-downloaded images
        new_images = [img for img in images if img["name"] not in existing_test]
        skipped = len(images) - len(new_images)
        total_skipped += skipped
        
        if skipped > 0:
            print(f"  Skipping {skipped} already-downloaded images")
        
        if not new_images:
            print(f"  All images already present. Nothing to download.")
            continue
        
        print(f"  Downloading {len(new_images)} test images...")
        
        # Download test images
        for i, img in enumerate(new_images):
            dest = test_class_dir / img["name"]
            if download_file(img["download_url"], str(dest)):
                total_downloaded += 1
            if (i + 1) % 20 == 0:
                print(f"    test: {i+1}/{len(new_images)}")
        
        # Report final counts
        final_test = len(list(test_class_dir.iterdir()))
        print(f"  Final: {final_test} test total")
    
    # Report on missing classes
    print("\n--- Classes not in PlantDoc (keeping existing data) ---")
    for cls in MISSING_CLASSES:
        test_count = len(list((TEST_DIR / cls).iterdir())) if (TEST_DIR / cls).exists() else 0
        print(f"  {cls}: {test_count} test (unchanged)")
    
    print("\n" + "=" * 60)
    print(f"Download complete!")
    print(f"  New images downloaded: {total_downloaded}")
    print(f"  Duplicates skipped:    {total_skipped}")
    print("=" * 60)
    
    # Final dataset summary
    print("\n--- Final Dataset Summary ---")
    total_test = 0
    for cls_dir in sorted(TEST_DIR.iterdir()):
        if cls_dir.is_dir():
            n = len([f for f in cls_dir.iterdir() if f.is_file()])
            total_test += n
    print(f"  Test:  {total_test} images")


if __name__ == "__main__":
    main()
