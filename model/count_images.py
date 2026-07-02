import os
for split in ["train", "val"]:
    print(f"\n=== {split} ===")
    total = 0
    root = os.path.join("data", split)
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if os.path.isdir(p):
            n = len([f for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))])
            total += n
            print(f"  {d}: {n}")
    print(f"  TOTAL: {total}")
