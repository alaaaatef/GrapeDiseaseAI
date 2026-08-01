import os
import shutil
from sklearn.model_selection import train_test_split

SEED = 42
TRAIN_RATIO = 0.8

source_dir = "data/processed/train"
validation_dir = "data/processed/validation"

os.makedirs(validation_dir, exist_ok=True)

classes = os.listdir(source_dir)

for cls in classes:

    class_path = os.path.join(source_dir, cls)

    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)

    train_imgs, val_imgs = train_test_split(
        images,
        train_size=TRAIN_RATIO,
        random_state=SEED,
        shuffle=True
    )

    os.makedirs(os.path.join(validation_dir, cls), exist_ok=True)

    for img in val_imgs:

        src = os.path.join(class_path, img)
        dst = os.path.join(validation_dir, cls, img)

        shutil.move(src, dst)

    print(
        f"{cls}: "
        f"Train={len(train_imgs)} | "
        f"Validation={len(val_imgs)}"
    )

print("\nDataset split completed successfully!")