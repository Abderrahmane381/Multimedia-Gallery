
import os
import json

from services.texture_features import extract_texture_features

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

mask_path = os.path.join(
    generated_folder,
    base_name + "_object.png"
)

if not os.path.exists(mask_path):
    print("Masque introuvable :", mask_path)
    raise SystemExit

# Analyse de toute l'image
global_features = extract_texture_features(image_path)

# Analyse de l'objet segmenté
object_features = extract_texture_features(
    image_path,
    mask_path=mask_path
)

print("\n========== TEXTURE IMAGE COMPLETE ==========")
print("Pixels analysés :", global_features["analyzed_pixels"])

print(json.dumps(
    global_features["glcm"],
    indent=4,
    ensure_ascii=False
))

print("\n========== TEXTURE OBJET SEGMENTE ==========")
print("Pixels objet :", object_features["analyzed_pixels"])
print("Pixels LBP analysés :", object_features["lbp_analyzed_pixels"])

print(json.dumps(
    object_features["glcm"],
    indent=4,
    ensure_ascii=False
))

print("\n========== COMPARAISON LBP ==========")
print("LBP global :")
print(global_features["lbp"]["histogram"])

print("\nLBP objet :")
print(object_features["lbp"]["histogram"])
