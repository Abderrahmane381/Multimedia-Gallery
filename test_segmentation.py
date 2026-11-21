
import os
from services.segmentation import (
    automatic_segmentation,
    save_segmentation,
    save_background
)

upload_folder = "uploads"
output_folder = "generated"

os.makedirs(output_folder, exist_ok=True)

images = [
    f for f in os.listdir(upload_folder)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
]

if not images:
    print("Aucune image disponible.")
else:
    image_path = os.path.join(upload_folder, images[0])

    image, mask = automatic_segmentation(image_path)

    filename = os.path.splitext(images[0])[0]

    object_path = os.path.join(
        output_folder, filename + "_object.png"
    )

    background_path = os.path.join(
        output_folder, filename + "_background.png"
    )

    save_segmentation(image, mask, object_path)
    save_background(image, mask, background_path)

    print("Segmentation terminée.")
    print("Objet :", object_path)
    print("Background :", background_path)
    print("Pixels détectés comme objet :", int((mask > 0).sum()))
