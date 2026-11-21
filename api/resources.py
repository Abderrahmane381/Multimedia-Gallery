
import os
import tempfile

from flask import request
from flask_restful import Resource

from services.color_features import extract_color_features
from services.texture_features import extract_texture_features
from services.feature_extraction import extract_all_features


def save_uploaded_file(file, directory, prefix):
    """Enregistre temporairement un fichier envoyé par l'API."""
    if file is None or not file.filename:
        raise ValueError("Le fichier image est obligatoire.")

    extension = os.path.splitext(file.filename)[1].lower()

    if extension not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}:
        raise ValueError("Format d'image non autorisé.")

    path = os.path.join(directory, prefix + extension)
    file.save(path)
    return path


class ColorFeaturesResource(Resource):
    def post(self):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                image_path = save_uploaded_file(
                    request.files.get("image"), temp_dir, "image"
                )

                mask_file = request.files.get("mask")
                mask_path = None

                if mask_file and mask_file.filename:
                    mask_path = save_uploaded_file(
                        mask_file, temp_dir, "mask"
                    )

                features = extract_color_features(
                    image_path,
                    mask_path=mask_path
                )

                return {
                    "success": True,
                    "descriptors": features
                }, 200

        except (ValueError, FileNotFoundError) as error:
            return {"success": False, "error": str(error)}, 400

        except Exception:
            return {
                "success": False,
                "error": "Erreur lors de l'extraction des descripteurs couleur."
            }, 500


class TextureFeaturesResource(Resource):
    def post(self):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                image_path = save_uploaded_file(
                    request.files.get("image"), temp_dir, "image"
                )

                mask_file = request.files.get("mask")
                mask_path = None

                if mask_file and mask_file.filename:
                    mask_path = save_uploaded_file(
                        mask_file, temp_dir, "mask"
                    )

                features = extract_texture_features(
                    image_path,
                    mask_path=mask_path
                )

                return {
                    "success": True,
                    "descriptors": features
                }, 200

        except (ValueError, FileNotFoundError) as error:
            return {"success": False, "error": str(error)}, 400

        except Exception:
            return {
                "success": False,
                "error": "Erreur lors de l'extraction des descripteurs texture."
            }, 500


class AllFeaturesResource(Resource):
    def post(self):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                image_path = save_uploaded_file(
                    request.files.get("image"), temp_dir, "image"
                )

                mode = request.form.get("mode", "global")

                mask_file = request.files.get("mask")
                mask_path = None

                if mask_file and mask_file.filename:
                    mask_path = save_uploaded_file(
                        mask_file, temp_dir, "mask"
                    )

                if mode in ("object", "background") and mask_path is None:
                    return {
                        "success": False,
                        "error": "Un masque est obligatoire pour ce mode."
                    }, 400

                features = extract_all_features(
                    image_path,
                    mode=mode,
                    object_mask_path=mask_path
                )

                return {
                    "success": True,
                    "descriptors": features
                }, 200

        except (ValueError, FileNotFoundError) as error:
            return {"success": False, "error": str(error)}, 400

        except Exception:
            return {
                "success": False,
                "error": "Erreur lors de l'extraction des descripteurs."
            }, 500
