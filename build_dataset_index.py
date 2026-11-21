
import os
import json
import numpy as np
from datetime import datetime

from services.feature_vector import extract_feature_vector


# Configuration
DATASET_PATH = os.path.join("dataset", "GHIM-10K")
OUTPUT_PATH = os.path.join(
    "database",
    "ghim10k_global_features.npz"
)
ERROR_PATH = os.path.join(
    "database",
    "ghim10k_index_errors.json"
)

EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Sauvegarde intermédiaire toutes les 100 images
CHECKPOINT_INTERVAL = 100


def save_index(vectors, paths, categories):
    """Enregistre l'index sur le disque."""

    os.makedirs("database", exist_ok=True)

    temp_path = OUTPUT_PATH + ".tmp.npz"

    np.savez_compressed(
        temp_path,
        vectors=np.asarray(vectors, dtype=np.float32),
        paths=np.asarray(paths),
        categories=np.asarray(categories)
    )

    os.replace(temp_path, OUTPUT_PATH)


def build_index():
    if not os.path.isdir(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset introuvable : {DATASET_PATH}"
        )

    vectors = []
    paths = []
    categories = []
    errors = []

    # Récupérer les catégories
    category_names = sorted(
        name for name in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, name))
    )

    image_paths = []

    for category in category_names:
        category_path = os.path.join(DATASET_PATH, category)

        for filename in sorted(os.listdir(category_path)):
            if filename.lower().endswith(EXTENSIONS):
                full_path = os.path.join(category_path, filename)
                image_paths.append((full_path, category))

    total = len(image_paths)

    print("=" * 55)
    print("INDEXATION DU DATASET GHIM-10K")
    print("=" * 55)
    print("Images trouvées :", total)
    print("Mode : global")
    print("Dimension des vecteurs : 83")
    print("=" * 55)

    for index, (image_path, category) in enumerate(image_paths, start=1):

        try:
            vector = extract_feature_vector(
                image_path=image_path,
                mode="global"
            )

            vectors.append(vector)
            paths.append(os.path.relpath(image_path).replace("\\", "/"))
            categories.append(category)

        except Exception as error:
            errors.append({
                "image": image_path,
                "error": str(error)
            })

        # Affichage de la progression
        if index % 10 == 0 or index == total:
            print(
                f"\rProgression : {index}/{total} "
                f"({index / total * 100:.1f}%) "
                f"| Réussies : {len(vectors)} "
                f"| Erreurs : {len(errors)}",
                end="",
                flush=True
            )

        # Sauvegarde intermédiaire
        if index % CHECKPOINT_INTERVAL == 0:
            save_index(vectors, paths, categories)

    # Sauvegarde finale
    save_index(vectors, paths, categories)

    # Enregistrer les erreurs éventuelles
    with open(ERROR_PATH, "w", encoding="utf-8") as file:
        json.dump(errors, file, ensure_ascii=False, indent=4)

    print("\n\n" + "=" * 55)
    print("INDEXATION TERMINEE")
    print("=" * 55)
    print("Images traitées :", len(vectors))
    print("Images ignorées :", len(errors))
    print("Dimension :", 83)
    print("Index enregistré :", OUTPUT_PATH)
    print("Erreurs enregistrées :", ERROR_PATH)

    print("Date :", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    build_index()
