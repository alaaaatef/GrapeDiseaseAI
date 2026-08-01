import os

print("===== TRAIN =====")
for cls in os.listdir("data/raw/train"):
    path = os.path.join("data/raw/train", cls)
    if os.path.isdir(path):
        print(f"{cls}: {len(os.listdir(path))}")

print("\n===== TEST =====")
for cls in os.listdir("data/raw/test"):
    path = os.path.join("data/raw/test", cls)
    if os.path.isdir(path):
        print(f"{cls}: {len(os.listdir(path))}")