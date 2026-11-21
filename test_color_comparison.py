
import os
import json

from services.color_features import extract_color_features

upload_folder = "uploads"
generated_folder = "generated"

images = [
    f for f in os.listdir(upload_folder)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
]

if not images:
    print("Aucune image trouvée dans uploads/")
    raise SystemExit

image_name = images[0]
image_path = os.path.join(upload_folder, image_name)

# Retrouver le PNG segmenté correspondant
base_name = os.path.splitext(image_name)[0]
mask_path = os.path.join(
    generated_folder,
    base_name + "_object.png"
)

if not os.path.exists(mask_path):
    print("Masque introuvable :", mask_path)
    raise SystemExit

# Extraction sur toute l'image
features_global = extract_color_features(image_path)

# Extraction uniquement sur l'objet
features_object = extract_color_features(
    image_path,
    mask_path=mask_path
)

print("\n========== IMAGE COMPLETE ==========")
print("Pixels analysés :", features_global["analyzed_pixels"])
print("Couleurs dominantes :")
print(json.dumps(
    features_global["dominant_colors"],
    indent=4,
    ensure_ascii=False
))

print("\n========== OBJET SANS BACKGROUND ==========")
print("Pixels analysés :", features_object["analyzed_pixels"])
print("Couleurs dominantes :")
print(json.dumps(
    features_object["dominant_colors"],
    indent=4,
    ensure_ascii=False
))
