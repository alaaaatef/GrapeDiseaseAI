import os
import hashlib

train_hashes = set()

for root, _, files in os.walk("data/raw/train"):
    for file in files:
        path = os.path.join(root, file)
        with open(path, "rb") as f:
            train_hashes.add(hashlib.md5(f.read()).hexdigest())

duplicates = 0

for root, _, files in os.walk("data/raw/test"):
    for file in files:
        path = os.path.join(root, file)
        with open(path, "rb") as f:
            h = hashlib.md5(f.read()).hexdigest()
            if h in train_hashes:
                duplicates += 1

print("Duplicate Images =", duplicates)