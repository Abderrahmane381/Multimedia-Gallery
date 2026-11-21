
import os
from services.feature_vector import extract_feature_vector
import numpy as np
# Trouver une image dans uploads
images = [
    f for f in os.listdir("uploads")
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

if not images:
    raise Exception("Aucune image trouvée dans uploads.")

image_path = os.path.join("uploads", images[0])

# Trouver son masque de segmentation
base_name = os.path.splitext(images[0])[0]
mask_path = os.path.join(
    "generated",
    base_name + "_object.png"
)

print("Image test :", image_path)

for mode in ("global", "object", "background"):
    print("\n==============================")
    print("MODE :", mode)
    print("==============================")

    if mode == "global":
        vector = extract_feature_vector(
            image_path,
            mode="global"
        )
    else:
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(
                "Masque objet introuvable : " + mask_path
            )

        vector = extract_feature_vector(
            image_path,
            mode=mode,
            object_mask_path=mask_path
        )

    print("Dimension :", len(vector))
    print("Type :", vector.dtype)
    print("Premières valeurs :", vector[:10])
    print("Valeurs finies :", np.all(np.isfinite(vector)))
