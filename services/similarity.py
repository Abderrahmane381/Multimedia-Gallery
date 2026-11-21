
import numpy as np


# Groupes de caractéristiques dans le vecteur de 83 dimensions
FEATURE_GROUPS = {
    "histograms": (0, 24),
    "moments": (24, 33),
    "dominant_colors": (33, 53),
    "glcm": (53, 57),
    "lbp": (57, 83)
}


def normalize_vector(vector):
    """
    Normalise séparément chaque groupe de caractéristiques
    afin d'éviter qu'un groupe domine les autres à cause
    de son échelle numérique.

    Chaque groupe est divisé par sa norme L2.
    """
    vector = np.asarray(vector, dtype=np.float32)

    if vector.shape != (83,):
        raise ValueError("Le vecteur doit contenir exactement 83 valeurs.")

    if not np.all(np.isfinite(vector)):
        raise ValueError("Le vecteur contient des valeurs invalides.")

    normalized = np.zeros_like(vector)

    for start, end in FEATURE_GROUPS.values():
        group = vector[start:end]
        norm = np.linalg.norm(group)

        if norm > 0:
            normalized[start:end] = group / norm

    return normalized


def euclidean_distance(vector1, vector2):
    """
    Calcule la distance euclidienne entre deux vecteurs normalisés.
    Une distance faible indique une plus grande similarité.
    """
    vector1 = normalize_vector(vector1)
    vector2 = normalize_vector(vector2)

    return float(np.linalg.norm(vector1 - vector2))


def weighted_distance(vector1, vector2, weights=None):
    """
    Calcule une distance euclidienne pondérée.

    weights : dictionnaire contenant un poids pour chaque groupe.
    Les poids par défaut sont identiques.
    """
    vector1 = normalize_vector(vector1)
    vector2 = normalize_vector(vector2)

    if weights is None:
        weights = {
            group: 1.0 for group in FEATURE_GROUPS
        }

    total_distance = 0.0
    total_weight = 0.0

    for group_name, (start, end) in FEATURE_GROUPS.items():
        weight = float(weights.get(group_name, 1.0))

        if not np.isfinite(weight) or weight < 0:
            raise ValueError("Les poids doivent être positifs ou nuls.")

        group_distance = np.linalg.norm(
            vector1[start:end] - vector2[start:end]
        )

        total_distance += weight * (group_distance ** 2)
        total_weight += weight

    if total_weight == 0:
        raise ValueError("La somme des poids doit être supérieure à zéro.")

    return float(np.sqrt(total_distance / total_weight))


def similarity_score(distance):
    """
    Transforme une distance en score de similarité compris
    entre 0 et 1. Plus le score est élevé, plus les images
    sont similaires.
    """
    return float(1.0 / (1.0 + distance))


def search_similar_images(query_vector, database_vectors, top_k=10,
                          weights=None):
    """
    Recherche les images les plus similaires à une image requête.

    Paramètres :
    - query_vector : vecteur de l'image requête.
    - database_vectors : dictionnaire {nom_image: vecteur}.
    - top_k : nombre maximal de résultats.
    - weights : poids des groupes de caractéristiques.

    Retour :
    Liste triée par distance croissante.
    """
    if top_k < 1:
        raise ValueError("top_k doit être supérieur ou égal à 1.")

    results = []

    for image_name, image_vector in database_vectors.items():
        distance = weighted_distance(
            query_vector,
            image_vector,
            weights
        )

        results.append({
            "image": image_name,
            "distance": distance,
            "similarity": similarity_score(distance)
        })

    results.sort(key=lambda item: item["distance"])

    return results[:top_k]
