
import os
import json

from services.feature_extraction import extract_all_features

upload_folder = "uploads"
generated_folder = "generated"

images = [
    f for f in os.listdir(upload_folder)
    if f.lower().endswith(
        (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    )
]

if not images:
    print("Aucune image trouvée.")
    raise SystemExit

image_name = images[0]
base_name = os.path.splitext(image_name)[0]

image_path = os.path.join(upload_folder, image_name)

object_mask_path = os.path.join(
    generated_folder,
    base_name + "_object.png"
)

if not os.path.exists(object_mask_path):
    print("Masque de segmentation introuvable.")
    raise SystemExit

for mode in ["global", "object", "background"]:

    print("\n================================")
    print("MODE :", mode)
    print("================================")

    features = extract_all_features(
        image_path,
        mode=mode,
        object_mask_path=object_mask_path
    )

    print("Pixels analysés :",
          features["color"]["analyzed_pixels"])

    print("\nCouleurs dominantes :")
    print(json.dumps(
        features["color"]["dominant_colors"],
        indent=4,
        ensure_ascii=False
    ))

    print("\nGLCM :")
    print(json.dumps(
        features["texture"]["glcm"],
        indent=4,
        ensure_ascii=False
    ))

    print("\nLBP :")
    print("Nombre de valeurs :",
          len(features["texture"]["lbp"]["histogram"]))
