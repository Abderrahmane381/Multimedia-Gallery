
import os

DATASET_PATH = os.path.join("dataset", "GHIM-10K")

extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

total_images = 0

print("VERIFICATION DU DATASET GHIM-10K")
print("=" * 45)

if not os.path.isdir(DATASET_PATH):
    raise FileNotFoundError(
        f"Dossier introuvable : {DATASET_PATH}"
    )

for category in sorted(os.listdir(DATASET_PATH)):
    category_path = os.path.join(DATASET_PATH, category)

    if not os.path.isdir(category_path):
        continue

    images = [
        file for file in os.listdir(category_path)
        if file.lower().endswith(extensions)
    ]

    count = len(images)
    total_images += count

    print(f"{category:20} : {count} images")

print("=" * 45)
print("TOTAL :", total_images, "images")
