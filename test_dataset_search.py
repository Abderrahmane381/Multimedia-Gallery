
import os
import numpy as np

from services.feature_vector import extract_feature_vector
from services.similarity import search_similar_images


# Chemins
INDEX_PATH = "database/ghim10k_global_features.npz"
QUERY_IMAGE = "uploads/5703572ec9214714be5742b5242b27d8.png"

# Vérifier les fichiers
if not os.path.isfile(INDEX_PATH):
    raise FileNotFoundError("Index du dataset introuvable.")

if not os.path.isfile(QUERY_IMAGE):
    raise FileNotFoundError("Image requête introuvable.")

print("Chargement de l'index...")

# Charger les données indexées
data = np.load(INDEX_PATH, allow_pickle=False)

vectors = data["vectors"]
paths = data["paths"]
categories = data["categories"]

print("Images indexées :", len(paths))
print("Dimensions :", vectors.shape)

# Extraire le vecteur de l'image requête
print("\nExtraction des caractéristiques de la requête...")

query_vector = extract_feature_vector(
    image_path=QUERY_IMAGE,
    mode="global"
)

# Préparer la base de recherche
database_vectors = {
    paths[i]: vectors[i]
    for i in range(len(paths))
}

# Effectuer la recherche
print("\nRecherche des 10 images les plus similaires...")

results = search_similar_images(
    query_vector=query_vector,
    database_vectors=database_vectors,
    top_k=10
)

# Associer chaque chemin à sa catégorie
category_by_path = {
    paths[i]: categories[i]
    for i in range(len(paths))
}

# Afficher les résultats
print("\n" + "=" * 65)
print("RESULTATS DE LA RECHERCHE CBIR")
print("=" * 65)

for rank, result in enumerate(results, start=1):
    image_path = result["image"]
    category = category_by_path[image_path]

    print(f"\nRésultat {rank}")
    print("Image :", image_path)
    print("Catégorie :", category)
    print("Distance :", round(result["distance"], 5))
    print("Similarité :", round(result["similarity"] * 100, 2), "%")

print("\n" + "=" * 65)
print("RECHERCHE TERMINEE")
print("=" * 65)
