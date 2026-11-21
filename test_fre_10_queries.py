
import numpy as np
import csv
import os

from services.relevance_feedback import fre_search
from services.similarity import search_similar_images


# ==================================================
# CONFIGURATION
# ==================================================

INDEX_PATH = "database/ghim10k_global_features.npz"

CATEGORIES = [
    "aircraft",
    "buildings",
    "butterfly",
    "cars",
    "dragon fly",
    "fireworks",
    "flowers",
    "hen",
    "horses",
    "insects",
    "motorcycles",
    "mountains",
    "sea shores",
    "ships",
    "sunset",
    "temples",
    "trees",
    "valleys",
    "walls",
    "yacht"
]

N_QUERIES = 10
ALPHA_VALUES = [0.25, 0.50, 0.75]
RANDOM_SEED = 42

CSV_PATH = "database/fre_20_categories_results.csv"


# ==================================================
# 1. CHARGEMENT DE L'INDEX
# ==================================================

data = np.load(INDEX_PATH, allow_pickle=False)

vectors = np.asarray(data["vectors"], dtype=np.float32)
paths = [str(p) for p in data["paths"]]
categories = [str(c) for c in data["categories"]]

category_by_path = dict(zip(paths, categories))

print("Nombre total d'images :", len(paths))
print("Dimension des vecteurs :", vectors.shape[1])
print("Nombre de catégories :", len(set(categories)))


# ==================================================
# 2. FONCTIONS UTILITAIRES
# ==================================================

def precision_at_k(results, target_category, k):
    selected = results[:k]

    relevant = sum(
        1 for item in selected
        if category_by_path[str(item["image"])] == target_category
    )

    return relevant / k


def rank_with_weights(query, database_vectors, database_paths, weights):

    differences = database_vectors - query

    distances = np.sqrt(
        np.sum(weights * differences ** 2, axis=1)
        / np.sum(weights)
    )

    order = np.argsort(distances)

    return [
        {
            "image": database_paths[i],
            "distance": float(distances[i])
        }
        for i in order
    ]


# ==================================================
# 3. SÉLECTION DES 10 REQUÊTES PAR CATÉGORIE
# ==================================================

rng = np.random.default_rng(RANDOM_SEED)

query_indices_by_category = {}

for category in CATEGORIES:

    indices = [
        i for i, c in enumerate(categories)
        if c == category
    ]

    if len(indices) < N_QUERIES:
        print("Catégorie ignorée :", category)
        continue

    selected = rng.choice(
        indices,
        size=N_QUERIES,
        replace=False
    )

    query_indices_by_category[category] = selected


# ==================================================
# 4. ÉVALUATION
# ==================================================

all_results = []

for category, query_indices in query_indices_by_category.items():

    print("\n" + "=" * 65)
    print("CATÉGORIE :", category)
    print("=" * 65)

    for query_number, query_index in enumerate(query_indices, start=1):

        query_path = paths[query_index]
        query_vector = vectors[query_index]

        # Exclure l'image requête de la base
        database_indices = [
            i for i in range(len(paths))
            if i != query_index
        ]

        db_vectors = vectors[database_indices]
        db_paths = [paths[i] for i in database_indices]

        database = {
            db_paths[i]: db_vectors[i]
            for i in range(len(db_paths))
        }

        # Recherche classique
        classical_results = search_similar_images(
            query_vector,
            database,
            top_k=25
        )

        # Simulation du feedback positif
        relevant_paths = [
            str(result["image"])
            for result in classical_results
            if category_by_path[str(result["image"])] == category
        ]

        # Sans exemple positif, FRE ne peut pas être appliqué
        if not relevant_paths:
            print(
                f"Requête {query_number}/10 ignorée : "
                "aucun exemple pertinent dans le top 25"
            )
            continue

        path_to_vector = {
            db_paths[i]: db_vectors[i]
            for i in range(len(db_paths))
        }

        relevant_vectors = np.asarray(
            [path_to_vector[p] for p in relevant_paths],
            dtype=np.float32
        )

        # Calcul des poids FRE
        fre_result = fre_search(
            relevant_vectors=relevant_vectors,
            database_vectors=db_vectors,
            database_paths=db_paths,
            top_k=25,
            query_vector=query_vector,
            relevant_paths=relevant_paths,
            alpha=0.25
        )

        initial_weights = fre_result["initial_weights"]
        updated_weights = fre_result["updated_weights"]
        updated_query = fre_result["updated_query"]

        # FRE individuel
        individual_results = rank_with_weights(
            query_vector,
            db_vectors,
            db_paths,
            initial_weights
        )

        row = {
            "category": category,
            "query": query_path,
            "feedback_count": len(relevant_paths),

            "classical_p10": precision_at_k(
                classical_results, category, 10
            ),
            "classical_p25": precision_at_k(
                classical_results, category, 25
            ),

            "individual_p10": precision_at_k(
                individual_results, category, 10
            ),
            "individual_p25": precision_at_k(
                individual_results, category, 25
            )
        }

        # Tester les différentes valeurs d'alpha
        for alpha in ALPHA_VALUES:

            weights = (
                alpha * initial_weights
                + (1 - alpha) * updated_weights
            )

            weights = weights / np.sum(weights)

            fre_results = rank_with_weights(
                updated_query,
                db_vectors,
                db_paths,
                weights
            )

            row[f"fre_{alpha}_p10"] = precision_at_k(
                fre_results, category, 10
            )

            row[f"fre_{alpha}_p25"] = precision_at_k(
                fre_results, category, 25
            )

        all_results.append(row)

        print(
            f"Requête {query_number}/10 | "
            f"Feedback : {len(relevant_paths)} | "
            f"Classique P@25 : "
            f"{row['classical_p25'] * 100:.1f}% | "
            f"FRE 0.25 P@25 : "
            f"{row['fre_0.25_p25'] * 100:.1f}%"
        )


# ==================================================
# 5. MOYENNES PAR CATÉGORIE
# ==================================================

methods = [
    ("Classique", "classical"),
    ("FRE individuel", "individual"),
    ("FRE alpha=0.25", "fre_0.25"),
    ("FRE alpha=0.50", "fre_0.5"),
    ("FRE alpha=0.75", "fre_0.75")
]

print("\n\n")
print("=" * 90)
print("RÉSULTATS MOYENS PAR CATÉGORIE")
print("=" * 90)

for category in CATEGORIES:

    rows = [
        r for r in all_results
        if r["category"] == category
    ]

    if not rows:
        print(f"\n{category} : aucune requête valide")
        continue

    print(
        f"\n{category.upper()} "
        f"({len(rows)} requêtes valides)"
    )

    for method_name, prefix in methods:

        p10 = np.mean([
            r[f"{prefix}_p10"] for r in rows
        ])

        p25 = np.mean([
            r[f"{prefix}_p25"] for r in rows
        ])

        print(
            f"{method_name:20s} "
            f"P@10 = {p10 * 100:6.2f}% | "
            f"P@25 = {p25 * 100:6.2f}%"
        )


# ==================================================
# 6. RÉSULTATS GLOBAUX
# ==================================================

print("\n" + "=" * 90)
print("RÉSULTATS GLOBAUX")
print("=" * 90)

if all_results:

    for method_name, prefix in methods:

        p10 = np.mean([
            r[f"{prefix}_p10"] for r in all_results
        ])

        p25 = np.mean([
            r[f"{prefix}_p25"] for r in all_results
        ])

        print(
            f"{method_name:20s} "
            f"P@10 = {p10 * 100:6.2f}% | "
            f"P@25 = {p25 * 100:6.2f}%"
        )


# ==================================================
# 7. EXPORT CSV
# ==================================================

if all_results:

    os.makedirs("database", exist_ok=True)

    fieldnames = list(all_results[0].keys())

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_results)

    print("\nFichier CSV enregistré :", CSV_PATH)

print("\nNombre de requêtes valides :", len(all_results))
print("Nombre de requêtes prévues :", len(CATEGORIES) * N_QUERIES)
print("ÉVALUATION TERMINÉE.")
