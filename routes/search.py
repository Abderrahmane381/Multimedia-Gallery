
import os
import uuid
import numpy as np

from flask import (
    Blueprint, render_template, request,
    redirect, url_for, flash, current_app,
    session, send_from_directory
)
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename

from models import Image, Category
from services.feature_vector import extract_feature_vector
from services.similarity import search_similar_images
from services.relevance_feedback import fre_search


search = Blueprint("search", __name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg","jfif", "bmp", "webp", "gif"}
RESULTS_COUNT = 25


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def load_dataset():
    """Charge les vecteurs pré-calculés de GHIM-10K."""

    index_path = os.path.join(
        current_app.root_path,
        "database",
        "ghim10k_global_features.npz"
    )

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            "Index GHIM-10K introuvable : " + index_path
        )

    with np.load(index_path, allow_pickle=False) as data:
        vectors = data["vectors"]
        paths = [
            str(p).replace("\\", "/")
            for p in data["paths"]
        ]
        categories = [
            str(c)
            for c in data["categories"]
        ]

    if not (len(vectors) == len(paths) == len(categories)):
        raise ValueError("Index GHIM-10K invalide.")

    database = {
        paths[i]: vectors[i]
        for i in range(len(paths))
    }

    metadata = {
        paths[i]: {
            "category": categories[i],
            "name": os.path.basename(paths[i])
        }
        for i in range(len(paths))
    }

    return database, metadata


def load_dataset_categories():
    """Retourne les catégories disponibles dans GHIM-10K."""

    _, metadata = load_dataset()

    return sorted({
        info["category"]
        for info in metadata.values()
    })


def load_gallery_categories():
    """Retourne les catégories créées par l'utilisateur connecté."""
    return sorted({
        category.name
        for category in Category.query.filter_by(user_id=current_user.id).all()
    })


def filter_category(database, metadata, selected_category):
    """Filtre une base (dataset ou galerie) selon la catégorie sélectionnée."""

    if not selected_category or selected_category == "all":
        return database, metadata

    filtered_database = {
        path: vector
        for path, vector in database.items()
        if metadata[path]["category"] == selected_category
    }

    filtered_metadata = {
        path: metadata[path]
        for path in filtered_database
    }

    return filtered_database, filtered_metadata


def load_user_gallery(selected_category="all"):
    """Extrait les vecteurs des images de la galerie personnelle."""

    images = Image.query.filter_by(
        user_id=current_user.id
    ).all()

    database = {}
    metadata = {}

    for image in images:

        if not os.path.exists(image.filepath):
            continue

        try:
            vector = extract_feature_vector(
                image.filepath,
                mode="global"
            )

            image_key = str(image.id)

            category_name = (
                image.category.name if image.category else "Sans catégorie"
            )
            if selected_category not in (None, "all") and category_name != selected_category:
                continue

            database[image_key] = vector

            metadata[image_key] = {
                "category": category_name,
                "name": image.original_name or image.filename
            }

        except Exception as error:
            current_app.logger.warning(
                "Extraction impossible pour %s : %s",
                image.filename,
                error
            )

    return database, metadata


def build_result_cards(results, source, metadata):
    """Prépare les résultats pour leur affichage."""

    cards = []

    for result in results:

        image_key = str(result["image"])
        info = metadata.get(image_key, {})

        if source == "dataset":

            image_url = url_for(
                "search.dataset_image",
                relative_path=image_key
            )

        else:

            image_id = int(image_key)

            image_record = Image.query.filter_by(
                id=image_id,
                user_id=current_user.id
            ).first_or_404()

            image_url = url_for(
                "gallery.uploaded_file",
                filename=image_record.filename
            )

        cards.append({
            "image_key": image_key,
            "image_url": image_url,
            "name": info.get(
                "name",
                os.path.basename(image_key)
            ),
            "category": info.get("category", "Inconnue"),
            "distance": round(float(result["distance"]), 5),
            "similarity": round(
                float(result["similarity"]) * 100, 2
            )
        })

    return cards


@search.route("/search", methods=["GET", "POST"])
@login_required
def image_search():

    dataset_categories = load_dataset_categories()
    gallery_categories = load_gallery_categories()
    categories = dataset_categories

    if request.method == "GET":
        return render_template(
            "search.html",
            results=None,
            source=None,
            categories=categories,
            dataset_categories=dataset_categories,
            gallery_categories=gallery_categories,
            selected_category="all",
            feedback_done=False
        )

    source = request.form.get("source")
    query_file = request.files.get("query_image")

    selected_category = request.form.get(
        "category",
        "all"
    )

    if source not in ("dataset", "gallery"):
        flash("Veuillez choisir une source de recherche.", "warning")
        return redirect(url_for("search.image_search"))

    if not query_file or not query_file.filename:
        flash("Veuillez sélectionner une image requête.", "warning")
        return redirect(url_for("search.image_search"))

    if not allowed_file(query_file.filename):
        flash("Format d'image non autorisé.", "danger")
        return redirect(url_for("search.image_search"))

    # Vérifier la catégorie selon la base choisie.
    categories = dataset_categories if source == "dataset" else gallery_categories
    if selected_category != "all" and selected_category not in categories:
        flash("Catégorie invalide.", "danger")
        return redirect(url_for("search.image_search"))

    # Enregistrer temporairement l'image requête.
    extension = secure_filename(
        query_file.filename
    ).rsplit(".", 1)[1].lower()

    query_filename = f"query_{uuid.uuid4().hex}.{extension}"

    query_path = os.path.join(
        current_app.config["UPLOAD_FOLDER"],
        query_filename
    )

    query_file.save(query_path)

    try:

        query_vector = extract_feature_vector(
            query_path,
            mode="global"
        )

        if source == "dataset":

            database, metadata = load_dataset()

            database, metadata = filter_category(
                database,
                metadata,
                selected_category
            )

        else:
            database, metadata = load_user_gallery(selected_category)

        if not database:
            raise ValueError(
                "Aucune image disponible pour cette recherche."
            )

        initial_results = search_similar_images(
            query_vector,
            database,
            top_k=RESULTS_COUNT
        )

        # Mémoriser les paramètres pour le feedback FRE.
        session["search_source"] = source
        session["search_query_filename"] = query_filename
        session["search_category"] = selected_category

        cards = build_result_cards(
            initial_results,
            source,
            metadata
        )

        return render_template(
            "search.html",
            results=cards,
            source=source,
            categories=categories,
            dataset_categories=dataset_categories,
            gallery_categories=gallery_categories,
            selected_category=selected_category,
            feedback_done=False
        )

    except Exception as error:

        if os.path.exists(query_path):
            os.remove(query_path)

        current_app.logger.exception("Erreur de recherche")

        flash(
            f"Erreur pendant la recherche : {error}",
            "danger"
        )

        return redirect(url_for("search.image_search"))


@search.route("/search/feedback", methods=["POST"])
@login_required
def apply_feedback():

    source = session.get("search_source")
    query_filename = session.get("search_query_filename")
    selected_category = session.get(
        "search_category",
        "all"
    )

    if source not in ("dataset", "gallery") or not query_filename:
        flash(
            "Veuillez effectuer une recherche avant le feedback.",
            "warning"
        )
        return redirect(url_for("search.image_search"))

    relevant_paths = request.form.getlist("relevant")

    if not relevant_paths:
        flash(
            "Sélectionnez au moins une image pertinente.",
            "warning"
        )
        return redirect(url_for("search.image_search"))

    query_path = os.path.join(
        current_app.config["UPLOAD_FOLDER"],
        query_filename
    )

    if not os.path.exists(query_path):
        flash(
            "L'image requête n'existe plus. Relancez la recherche.",
            "danger"
        )
        return redirect(url_for("search.image_search"))

    try:

        query_vector = extract_feature_vector(
            query_path,
            mode="global"
        )

        dataset_categories = load_dataset_categories()
        gallery_categories = load_gallery_categories()
        categories = dataset_categories if source == "dataset" else gallery_categories

        if source == "dataset":

            database, metadata = load_dataset()

            # Garder exactement le même filtre que la recherche initiale.
            database, metadata = filter_category(
                database,
                metadata,
                selected_category
            )

        else:
            database, metadata = load_user_gallery(selected_category)

        if not database:
            raise ValueError("La base de recherche est vide.")

        # Vérifier les images sélectionnées.
        relevant_paths = [
            path
            for path in relevant_paths
            if path in database
        ]

        if not relevant_paths:
            raise ValueError(
                "Les images sélectionnées ne sont plus disponibles."
            )

        database_paths = list(database.keys())

        database_vectors = np.asarray(
            [database[path] for path in database_paths],
            dtype=np.float32
        )

        path_to_vector = {
            path: database[path]
            for path in database_paths
        }

        relevant_vectors = np.asarray(
            [path_to_vector[path] for path in relevant_paths],
            dtype=np.float32
        )

        # Nouvelle recherche avec FRE.
        fre_result = fre_search(
            relevant_vectors=relevant_vectors,
            database_vectors=database_vectors,
            database_paths=database_paths,
            top_k=RESULTS_COUNT,
            query_vector=query_vector,
            relevant_paths=relevant_paths
        )

        cards = build_result_cards(
            fre_result["results"],
            source,
            metadata
        )

        return render_template(
            "search.html",
            results=cards,
            source=source,
            categories=categories,
            dataset_categories=dataset_categories,
            gallery_categories=gallery_categories,
            selected_category=selected_category,
            feedback_done=True,
            relevant_count=len(relevant_paths)
        )

    except Exception as error:

        current_app.logger.exception("Erreur FRE")

        flash(
            f"Erreur pendant le feedback : {error}",
            "danger"
        )

        return redirect(url_for("search.image_search"))


@search.route("/search/dataset-image/<path:relative_path>")
@login_required
def dataset_image(relative_path):
    """Affiche une image du dataset GHIM-10K."""

    return send_from_directory(
        current_app.root_path,
        relative_path
    )
