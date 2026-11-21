
import numpy as np

from services.similarity import (
    euclidean_distance,
    weighted_distance,
    similarity_score,
    search_similar_images
)

# Création de trois vecteurs de test
query = np.random.rand(83).astype(np.float32)

image1 = query.copy()
image2 = np.random.rand(83).astype(np.float32)
image3 = (query + 0.1).astype(np.float32)

# Comparaison de deux vecteurs
print("Distance euclidienne :", euclidean_distance(query, image1))
print("Distance pondérée :", weighted_distance(query, image2))
print("Score de similarité :", similarity_score(0.5))

# Recherche des images similaires
database = {
    "image1.jpg": image1,
    "image2.jpg": image2,
    "image3.jpg": image3
}

results = search_similar_images(
    query_vector=query,
    database_vectors=database,
    top_k=3
)

print("\nClassement des images :")

for result in results:
    print(result)
