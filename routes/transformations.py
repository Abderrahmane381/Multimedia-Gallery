
import os
import uuid

from PIL import Image as PILImage
from flask import (
    Blueprint, render_template, request, redirect,
    url_for, flash, current_app, send_from_directory
)
from flask_login import login_required, current_user

from models import Image

transformations = Blueprint("transformations", __name__)


def get_user_image(image_id):
    return Image.query.filter_by(
        id=image_id,
        user_id=current_user.id
    ).first_or_404()


@transformations.route("/image/<int:image_id>")
@login_required
def image_details(image_id):
    image = get_user_image(image_id)

    result_filename = request.args.get("result")
    result_url = None

    if result_filename:
        expected_prefix = f"img_{image.id}_"

        if result_filename.startswith(expected_prefix):
            result_path = os.path.join(
                current_app.config["GENERATED_FOLDER"],
                result_filename
            )

            if os.path.isfile(result_path):
                result_url = url_for(
                    "transformations.generated_file",
                    filename=result_filename
                )

    return render_template(
        "image_details.html",
        image=image,
        result_url=result_url
    )


@transformations.route("/transform/<int:image_id>", methods=["POST"])
@login_required
def transform_image(image_id):
    image = get_user_image(image_id)

    operation = request.form.get("operation")

    source_path = os.path.join(
        current_app.config["UPLOAD_FOLDER"],
        image.filename
    )

    if not os.path.isfile(source_path):
        flash("Le fichier original est introuvable.", "danger")
        return redirect(url_for(
            "transformations.image_details",
            image_id=image.id
        ))

    try:
        with PILImage.open(source_path) as original:
            original.load()
            original = original.convert("RGB")

            if operation == "resize":
                width = int(request.form.get("width", "0"))
                height = int(request.form.get("height", "0"))

                if not (1 <= width <= 5000 and 1 <= height <= 5000):
                    raise ValueError("Dimensions invalides (1 à 5000 pixels).")

                transformed = original.resize((width, height))

            elif operation == "crop":
                x = int(request.form.get("x", "0"))
                y = int(request.form.get("y", "0"))
                width = int(request.form.get("width", "0"))
                height = int(request.form.get("height", "0"))

                if x < 0 or y < 0 or width <= 0 or height <= 0:
                    raise ValueError("Coordonnées ou dimensions invalides.")

                if x + width > original.width or y + height > original.height:
                    raise ValueError("La zone dépasse les limites de l'image.")

                transformed = original.crop((
                    x, y, x + width, y + height
                ))

            else:
                raise ValueError("Opération inconnue.")

            filename = f"img_{image.id}_{uuid.uuid4().hex}.jpg"
            output_path = os.path.join(
                current_app.config["GENERATED_FOLDER"],
                filename
            )

            transformed.save(output_path, format="JPEG", quality=95)

        flash("Transformation effectuée avec succès.", "success")

        return redirect(url_for(
            "transformations.image_details",
            image_id=image.id,
            result=filename
        ))

    except (ValueError, TypeError) as error:
        flash(str(error), "danger")
    except Exception:
        flash("Impossible de transformer cette image.", "danger")

    return redirect(url_for(
        "transformations.image_details",
        image_id=image.id
    ))


@transformations.route("/generated/<filename>")
@login_required
def generated_file(filename):
    if not filename.startswith("img_"):
        return "Fichier invalide", 404

    try:
        image_id = int(filename.split("_")[1])
    except (ValueError, IndexError):
        return "Fichier invalide", 404

    get_user_image(image_id)

    return send_from_directory(
        current_app.config["GENERATED_FOLDER"],
        filename
    )
