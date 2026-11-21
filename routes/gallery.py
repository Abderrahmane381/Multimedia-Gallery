
import os
import uuid

from flask import (
    Blueprint, render_template, request, redirect,
    url_for, flash, current_app, send_from_directory, abort
)
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename

from extensions import db
from models import Image, Category

gallery = Blueprint("gallery", __name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg","jfif", "gif", "bmp", "webp"}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )



@gallery.route("/gallery")
@login_required
def show_gallery():
    images = Image.query.filter_by(
        user_id=current_user.id
    ).order_by(Image.upload_date.desc()).all()

    categories = Category.query.filter_by(
        user_id=current_user.id
    ).all()

    return render_template(
        "gallery.html",
        images=images,
        categories=categories
    )




@gallery.route("/upload", methods=["POST"])
@login_required
def upload_images():
    # Récupérer les images sélectionnées
    files = request.files.getlist("photos")

    # Récupérer la catégorie choisie dans le formulaire
    category_id = request.form.get("category_id")

    # Vérifier la catégorie
    category = None

    if category_id:
        category = Category.query.filter_by(
            id=category_id,
            user_id=current_user.id
        ).first()

        if category is None:
            flash("Catégorie invalide.", "danger")
            return redirect(url_for("gallery.show_gallery"))

    uploaded_count = 0

    for file in files:
        if not file or not file.filename:
            continue

        if not allowed_file(file.filename):
            continue

        original_name = secure_filename(file.filename)
        extension = original_name.rsplit(".", 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{extension}"

        filepath = os.path.join(
            current_app.config["UPLOAD_FOLDER"], filename
        )

        file.save(filepath)

        # Enregistrer l'image avec sa catégorie
        image = Image(
            filename=filename,
            original_name=original_name,
            filepath=filepath,
            user_id=current_user.id,
            category_id=category.id if category else None
        )

        db.session.add(image)
        uploaded_count += 1

    if uploaded_count:
        db.session.commit()
        flash(f"{uploaded_count} image(s) importée(s).", "success")
    else:
        flash("Aucune image valide sélectionnée.", "warning")

    return redirect(url_for("gallery.show_gallery"))



@gallery.route("/uploads/<filename>")
@login_required
def uploaded_file(filename):
    image = Image.query.filter_by(
        filename=filename,
        user_id=current_user.id
    ).first_or_404()

    return send_from_directory(
        current_app.config["UPLOAD_FOLDER"],
        image.filename
    )


@gallery.route("/download/<int:image_id>")
@login_required
def download_image(image_id):
    image = Image.query.filter_by(
        id=image_id, user_id=current_user.id
    ).first_or_404()

    return send_from_directory(
        current_app.config["UPLOAD_FOLDER"],
        image.filename,
        as_attachment=True,
        download_name=image.original_name or image.filename
    )


@gallery.route("/delete/<int:image_id>", methods=["POST"])
@login_required
def delete_image(image_id):
    image = Image.query.filter_by(
        id=image_id, user_id=current_user.id
    ).first_or_404()

    filepath = os.path.join(
        current_app.config["UPLOAD_FOLDER"], image.filename
    )

    if os.path.exists(filepath):
        os.remove(filepath)

    db.session.delete(image)
    db.session.commit()

    flash("Image supprimée.", "success")
    return redirect(url_for("gallery.show_gallery"))
