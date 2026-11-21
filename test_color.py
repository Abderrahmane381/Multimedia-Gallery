
import os
import json

from services.color_features import extract_color_features

# Chercher automatiquement une image dans uploads/
upload_folder = "uploads"

images = [
    filename for filename in os.listdir(upload_folder)
    if filename.lower().endswith(
        (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    )
]

if not images:
    print("Aucune image trouvée dans uploads/")
else:
    image_path = os.path.join(upload_folder, images[0])

    print("Image testée :", images[0])

    features = extract_color_features(image_path)

    print("\nDESCRIPTEURS DE COULEUR")
    print(json.dumps(features, indent=4, ensure_ascii=False))
