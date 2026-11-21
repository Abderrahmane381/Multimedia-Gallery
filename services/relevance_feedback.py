
import numpy as np
from services.similarity import normalize_vector

EPSILON = 0.0001
EXPECTED_DIMENSION = 83


def _validate_vectors(vectors, name="vecteurs"):
    """Vérifie la forme et les valeurs d'une matrice de vecteurs."""
    vectors = np.asarray(vectors, dtype=np.float64)

    if vectors.ndim != 2:
        raise ValueError(f"{name} doivent former une matrice 2D.")

    if vectors.shape[1] != EXPECTED_DIMENSION:
        raise ValueError(f"Chaque vecteur doit avoir 83 dimensions.")

    if vectors.shape[0] == 0:
        raise ValueError(f"Aucun {name} fourni.")

    if not np.all(np.isfinite(vectors)):
        raise ValueError(f"{name} contiennent des valeurs invalides.")

    return vectors


def _validate_vector(vector, name="vecteur"):
    """Vérifie un vecteur individuel de 83 dimensions."""
    vector = np.asarray(vector, dtype=np.float64)

    if vector.shape != (EXPECTED_DIMENSION,):
        raise ValueError(f"{name} doit avoir 83 dimensions.")

    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contient des valeurs invalides.")

    return vector


def estimate_individual_feature_relevance(
    query_vector,
    database_vectors,
    relevant_paths,
    database_paths,
    top_k=25
):
    """
    Estime la pertinence de chaque composante individuellement.

    Pour chaque composante :
    1. Compare la requête aux images de la base sur cette composante seule.
    2. Sélectionne les top_k plus proches images.
    3. Compte combien d'images pertinentes sont retrouvées.
    4. Calcule un poids proportionnel à ce nombre.

    Selon la première étape FRE de l'article :
        poids_i = nombre_images_pertinentes_retrouvees_i
                  / somme_des_nombres

    Retourne les nombres de résultats pertinents et les poids normalisés.
    """
    query_vector = _validate_vector(query_vector, "La requête")
    database_vectors = _validate_vectors(
        database_vectors, "La base de données"
    )

    if len(database_paths) != len(database_vectors):
        raise ValueError("Le nombre de chemins et de vecteurs diffère.")

    if top_k < 1:
        raise ValueError("top_k doit être supérieur ou égal à 1.")

    relevant_paths = {str(path) for path in relevant_paths}
    database_paths = [str(path) for path in database_paths]

    number_to_retrieve = min(top_k, len(database_vectors))

    relevance_counts = np.zeros(EXPECTED_DIMENSION, dtype=np.float64)

    # Tester chaque composante séparément
    for feature_index in range(EXPECTED_DIMENSION):

        distances = np.abs(
            database_vectors[:, feature_index]
            - query_vector[feature_index]
        )

        nearest_indices = np.argsort(distances)[:number_to_retrieve]

        # Compter les images pertinentes parmi les voisins retrouvés
        for index in nearest_indices:
            if database_paths[index] in relevant_paths:
                relevance_counts[feature_index] += 1

    total_relevant = np.sum(relevance_counts)

    # Normaliser les poids
    if total_relevant > 0:
        initial_weights = relevance_counts / total_relevant
    else:
        # Aucun résultat pertinent : poids uniformes
        initial_weights = np.ones(EXPECTED_DIMENSION) / EXPECTED_DIMENSION

    return {
        "relevance_counts": relevance_counts.astype(np.int32),
        "weights": initial_weights.astype(np.float32)
    }


def compute_fre_weights(
    relevant_vectors,
    retrieved_vectors,
    epsilon=EPSILON
):
    """
    Calcule les poids FRE mis à jour selon l'article.

    Formule :
        w_i(k+1) = (epsilon + sigma_Nr,i)
                   / (epsilon + sigma_rel,i)

    sigma_Nr,i  : écart-type de la composante i parmi les images récupérées.
    sigma_rel,i : écart-type de la composante i parmi les images pertinentes.

    Les poids sont ensuite normalisés.
    """
    relevant_vectors = _validate_vectors(
        relevant_vectors, "Les vecteurs pertinents"
    )
    retrieved_vectors = _validate_vectors(
        retrieved_vectors, "Les vecteurs récupérés"
    )

    if epsilon <= 0:
        raise ValueError("epsilon doit être positif.")

    std_retrieved = np.std(retrieved_vectors, axis=0)
    std_relevant = np.std(relevant_vectors, axis=0)

    weights = (
        (epsilon + std_retrieved)
        / (epsilon + std_relevant)
    )

    total = np.sum(weights)

    if not np.isfinite(total) or total <= 0:
        raise ValueError("Impossible de normaliser les poids FRE.")

    weights = weights / total

    return weights.astype(np.float32)


def update_fre_query(relevant_vectors):
    """
    Reformule la requête en calculant la moyenne
    des vecteurs des images pertinentes.
    """
    relevant_vectors = _validate_vectors(
        relevant_vectors, "Les vecteurs pertinents"
    )

    return np.mean(relevant_vectors, axis=0).astype(np.float32)


def fre_distance(vector1, vector2, weights):
    """
    Distance euclidienne pondérée :

        Dist(X,Y) = sqrt(
            somme(w_i * (x_i-y_i)^2) / somme(w_i)
        )
    """
    vector1 = _validate_vector(vector1, "Le premier vecteur")
    vector2 = _validate_vector(vector2, "Le deuxième vecteur")

    weights = np.asarray(weights, dtype=np.float64)

    if weights.shape != (EXPECTED_DIMENSION,):
        raise ValueError("Il faut 83 poids FRE.")

    if not np.all(np.isfinite(weights)):
        raise ValueError("Les poids contiennent des valeurs invalides.")

    if np.any(weights < 0) or np.sum(weights) <= 0:
        raise ValueError("Les poids doivent être positifs.")

    squared_differences = (vector1 - vector2) ** 2

    distance_squared = np.sum(
        weights * squared_differences
    ) / np.sum(weights)

    return float(np.sqrt(distance_squared))


def fre_similarity_score(distance):
    """Convertit une distance en score de similarité entre 0 et 1."""
    if distance < 0 or not np.isfinite(distance):
        raise ValueError("Distance invalide.")

    return float(1.0 / (1.0 + distance))


def fre_search(
    relevant_vectors,
    database_vectors,
    database_paths,
    top_k=10,
    epsilon=EPSILON,
    query_vector=None,
    relevant_paths=None,
    alpha=0.75
):
    """
    Recherche FRE après retour de pertinence.

    Étapes :
    1. Estimation individuelle des 83 composantes.
    2. Récupération des images les plus proches de la requête.
    3. Calcul des écarts-types des images récupérées et pertinentes.
    4. Mise à jour des poids FRE.
    5. Reformulation de la requête par la moyenne des images pertinentes.
    6. Classement de la base avec la distance pondérée.

    query_vector :
        Vecteur de la requête initiale. Si absent, utilise la moyenne
        des vecteurs pertinents.

    relevant_paths :
        Chemins des images marquées pertinentes par l'utilisateur.
        Si absent, les images pertinentes sont retrouvées par comparaison
        exacte de leurs vecteurs avec ceux de la base.
    """
    relevant_vectors = _validate_vectors(
        relevant_vectors, "Les vecteurs pertinents"
    )
    database_vectors = _validate_vectors(
        database_vectors, "Les vecteurs de la base"
    )

    database_paths = [str(path) for path in database_paths]

    if len(database_paths) != len(database_vectors):
        raise ValueError("Nombre de chemins et de vecteurs différent.")

    if top_k < 1:
        raise ValueError("top_k doit être supérieur ou égal à 1.")

    if query_vector is None:
        query_vector = update_fre_query(relevant_vectors)
    else:
        query_vector = _validate_vector(
            query_vector, "La requête initiale"
        )

    # Identifier les chemins pertinents si non fournis
    if relevant_paths is None:
        relevant_paths = set()

        for relevant_vector in relevant_vectors:
            matches = np.where(
                np.all(
                    np.isclose(
                        database_vectors,
                        relevant_vector,
                        rtol=1e-7,
                        atol=1e-9
                    ),
                    axis=1
                )
            )[0]

            for index in matches:
                relevant_paths.add(database_paths[index])

    else:
        relevant_paths = {str(path) for path in relevant_paths}

    # Étape 1 : évaluer chaque composante individuellement
    individual_estimation = estimate_individual_feature_relevance(
        query_vector=query_vector,
        database_vectors=database_vectors,
        relevant_paths=relevant_paths,
        database_paths=database_paths,
        top_k=max(top_k, 25)
    )

    initial_weights = individual_estimation["weights"]

    # Étape 2 : récupérer les images les plus proches
    # Utiliser la même normalisation que la recherche classique

    normalized_query = normalize_vector(query_vector)

    normalized_database = np.vstack([
        normalize_vector(vector)
        for vector in database_vectors
    ])

    initial_distances = np.linalg.norm(
        normalized_database - normalized_query,
        axis=1
    )

    retrieved_count = min(max(top_k, 25), len(database_vectors))
    retrieved_indices = np.argsort(initial_distances)[:retrieved_count]

    retrieved_vectors = database_vectors[retrieved_indices]

    # Étape 3 : poids mis à jour par les écarts-types
    updated_weights = compute_fre_weights(
        relevant_vectors=relevant_vectors,
        retrieved_vectors=retrieved_vectors,
        epsilon=epsilon
    )

   # Combinaison additive des deux estimations
    if not 0 <= alpha <= 1:
        raise ValueError("alpha doit être compris entre 0 et 1.")

    weights = (
        alpha * initial_weights
        + (1 - alpha) * updated_weights
    )   

    weights = weights / np.sum(weights)

    print("\n===== DIAGNOSTIC DES POIDS FRE =====")

    print("Poids initiaux nuls :",
      np.sum(initial_weights == 0))

    print("Poids mis à jour nuls :",
      np.sum(updated_weights == 0))

    print("Poids combinés nuls :",
      np.sum(weights == 0))

    print("Poids initiaux - min/max :",
      np.min(initial_weights), np.max(initial_weights))

    print("Poids mis à jour - min/max :",
      np.min(updated_weights), np.max(updated_weights))

    print("Poids combinés - min/max :",
      np.min(weights), np.max(weights))


    weights = weights.astype(np.float32)

    # Étape 4 : reformulation de la requête
    updated_query = update_fre_query(relevant_vectors)

    # Étape 5 : nouvelle recherche dans toute la base
    results = []

    for index, vector in enumerate(database_vectors):
        distance = fre_distance(
            updated_query,
            vector,
            weights
        )

        results.append({
            "image": database_paths[index],
            "distance": distance,
            "similarity": fre_similarity_score(distance)
        })

    results.sort(key=lambda item: item["distance"])

    return {
        "updated_query": updated_query,
        "initial_weights": initial_weights,
        "updated_weights": updated_weights,
        "weights": weights,
        "relevance_counts": individual_estimation["relevance_counts"],
        "results": results[:top_k]
    }
