
import os
import numpy as np

from services.feature_vector import extract_feature_vector
from services.similarity import search_similar_images
from services.relevance_feedback import fre_search, fre_distance


# ==================================================
# CONFIGURATION
# ==================================================

INDEX_PATH = "database/ghim10k_global_features.npz"

TARGET_CATEGORIES = [
    "aircraft",
    "yacht",
    "ships",
    "fireworks"
]

K_VALUES = [10, 25]

ALPHA_VALUES = [0.25, 0.50, 0.75]

TOP_K = 25


# ==================================================
# 1. CHARGEMENT DU DATASET
# ==================================================

data = np.load(INDEX_PATH, allow_pickle=False)

vectors = data["vectors"]

paths = [
    str(path).replace("\\", "/")
    for path in data["paths"]
]

categories = [
    str(category)
    for category in data["categories"]
]

print("=" * 70)
print("DATASET GHIM-10K")
print("=" * 70)

print("Nombre d'images :", len(paths))
print("Dimension des vecteurs :", vectors.shape[1])
print("Nombre de catégories :", len(set(categories)))

available_categories = sorted(set(categories))

print("\nCatégories disponibles :")

for category in available_categories:
    print("-", category)


# ==================================================
# 2. PREPARATION DES DONNEES
# ==================================================

category_by_path = {
    paths[i]: categories[i]
    for i in range(len(paths))
}

path_to_vector = {
    paths[i]: vectors[i]
    for i in range(len(paths))
}


def find_query_image(category):
    """
    Choisit automatiquement une image existante
    dans la catégorie demandée.
    """

    for path, image_category in zip(paths, categories):

        if image_category != category:
            continue

        if os.path.isfile(path):
            return path

    return None


def precision_at_k(results, k, target_category):
    """
    Calcule Precision@K selon la catégorie réelle.
    """

    selected_results = results[:k]

    relevant_count = sum(
        1
        for result in selected_results
        if category_by_path[str(result["image"])]
        == target_category
    )

    return relevant_count, relevant_count / k


# ==================================================
# 3. EXPERIMENTATION SUR CHAQUE CATEGORIE
# ==================================================

all_experiment_results = {}

for target_category in TARGET_CATEGORIES:

    print("\n\n")
    print("=" * 70)
    print("CATEGORIE :", target_category.upper())
    print("=" * 70)

    # --------------------------------------------------
    # 3.1 Vérifier la catégorie
    # --------------------------------------------------

    if target_category not in available_categories:

        print("Catégorie inexistante dans le dataset.")
        continue

    # --------------------------------------------------
    # 3.2 Choisir une image requête
    # --------------------------------------------------

    query_image = find_query_image(target_category)

    if query_image is None:

        print("Aucune image accessible pour cette catégorie.")
        print("Vérifie les chemins des images dans l'index.")
        continue

    print("Image requête :", query_image)

    query_vector = extract_feature_vector(
        query_image,
        mode="global"
    )

    # --------------------------------------------------
    # 3.3 Exclure l'image requête de la base
    # --------------------------------------------------

    database_indices = [
        i for i, path in enumerate(paths)
        if path != query_image
    ]

    database_paths = [
        paths[i] for i in database_indices
    ]

    database_vectors = vectors[database_indices]

    database = {
        database_paths[i]: database_vectors[i]
        for i in range(len(database_paths))
    }

    # --------------------------------------------------
    # 3.4 Recherche classique
    # --------------------------------------------------

    initial_results = search_similar_images(
        query_vector,
        database,
        top_k=TOP_K
    )

    print("\nRecherche classique terminée.")

    # --------------------------------------------------
    # 3.5 Simulation du feedback positif
    # --------------------------------------------------

    relevant_paths = [
        result["image"]
        for result in initial_results
        if category_by_path[str(result["image"])]
        == target_category
    ]

    if not relevant_paths:

        print(
            "Aucune image de la catégorie cible "
            "dans les 25 premiers résultats."
        )

        print("Expérience FRE ignorée pour cette catégorie.")
        continue

    relevant_vectors = np.asarray(
        [
            path_to_vector[path]
            for path in relevant_paths
        ],
        dtype=np.float32
    )

    print(
        "Images pertinentes sélectionnées :",
        len(relevant_paths)
    )

    # --------------------------------------------------
    # 3.6 Recherche FRE pour chaque alpha
    # --------------------------------------------------

    fre_results_by_alpha = {}

    for alpha in ALPHA_VALUES:

        print(f"\nRecherche FRE avec alpha = {alpha}")

        fre_result = fre_search(
            relevant_vectors=relevant_vectors,
            database_vectors=database_vectors,
            database_paths=database_paths,
            top_k=TOP_K,
            query_vector=query_vector,
            relevant_paths=relevant_paths,
            alpha=alpha
        )

        fre_results_by_alpha[alpha] = fre_result

    # --------------------------------------------------
    # 3.7 Estimation individuelle
    # --------------------------------------------------

    individual_weights = (
        fre_results_by_alpha[ALPHA_VALUES[0]]
        ["initial_weights"]
    )

    individual_results = []

    for i, vector in enumerate(database_vectors):

        distance = fre_distance(
            query_vector,
            vector,
            individual_weights
        )

        individual_results.append({
            "image": database_paths[i],
            "distance": distance
        })

    individual_results.sort(
        key=lambda item: item["distance"]
    )

    # --------------------------------------------------
    # 3.8 Comparaison des méthodes
    # --------------------------------------------------

    print("\n" + "-" * 70)
    print("RESULTATS :", target_category.upper())
    print("-" * 70)

    methods = {
        "Recherche classique": initial_results,
        "FRE individuel": individual_results
    }

    for name, results in methods.items():

        print("\n", name)

        for k in K_VALUES:

            count, precision = precision_at_k(
                results,
                k,
                target_category
            )

            print(
                f"Precision@{k} : "
                f"{count}/{k} = {precision * 100:.2f}%"
            )

    for alpha, result in fre_results_by_alpha.items():

        print(f"\nFRE complet - alpha = {alpha}")

        for k in K_VALUES:

            count, precision = precision_at_k(
                result["results"],
                k,
                target_category
            )

            print(
                f"Precision@{k} : "
                f"{count}/{k} = {precision * 100:.2f}%"
            )

    # --------------------------------------------------
    # 3.9 Identifier le meilleur alpha
    # --------------------------------------------------

    best_alpha = max(
        ALPHA_VALUES,
        key=lambda alpha: (
            precision_at_k(
                fre_results_by_alpha[alpha]["results"],
                25,
                target_category
            )[1],

            precision_at_k(
                fre_results_by_alpha[alpha]["results"],
                10,
                target_category
            )[1]
        )
    )

    best_result = fre_results_by_alpha[best_alpha]

    p10_count, p10 = precision_at_k(
        best_result["results"],
        10,
        target_category
    )

    p25_count, p25 = precision_at_k(
        best_result["results"],
        25,
        target_category
    )

    print("\nMeilleur alpha :", best_alpha)

    print(
        f"Meilleure Precision@10 : "
        f"{p10_count}/10 = {p10 * 100:.2f}%"
    )

    print(
        f"Meilleure Precision@25 : "
        f"{p25_count}/25 = {p25 * 100:.2f}%"
    )

    # --------------------------------------------------
    # 3.10 Enregistrer les résultats
    # --------------------------------------------------

    all_experiment_results[target_category] = {
        "classical": initial_results,
        "individual": individual_results,
        "fre": fre_results_by_alpha,
        "best_alpha": best_alpha,
        "precision10": p10,
        "precision25": p25
    }


# ==================================================
# 4. TABLEAU RECAPITULATIF
# ==================================================

print("\n\n")
print("=" * 80)
print("TABLEAU RECAPITULATIF DES EXPERIENCES")
print("=" * 80)

print(
    f"{'Categorie':<15}"
    f"{'P@10 classique':<18}"
    f"{'P@25 classique':<18}"
    f"{'Meilleur alpha':<16}"
    f"{'P@25 FRE':<12}"
)

for category, experiment in all_experiment_results.items():

    classical_p10 = precision_at_k(
        experiment["classical"], 10, category
    )[1]

    classical_p25 = precision_at_k(
        experiment["classical"], 25, category
    )[1]

    print(
        f"{category:<15}"
        f"{classical_p10 * 100:<18.2f}"
        f"{classical_p25 * 100:<18.2f}"
        f"{experiment['best_alpha']:<16}"
        f"{experiment['precision25'] * 100:.2f}%"
    )


print("\nNombre de catégories testées :",
      len(all_experiment_results))

print("\nEXPERIMENTATION TERMINEE.")
