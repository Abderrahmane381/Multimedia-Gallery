
from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user
from extensions import db
from models import Category

categories = Blueprint("categories", __name__)


@categories.route("/categories")
@login_required
def index():
    user_categories = Category.query.filter_by(
        user_id=current_user.id
    ).all()

    return render_template(
        "categories.html",
        categories=user_categories
    )


@categories.route("/categories/create", methods=["POST"])
@login_required
def create_category():
    name = request.form.get("name", "").strip()
    parent_id = request.form.get("parent_id")

    if not name:
        flash("Le nom de la catégorie est obligatoire.", "danger")
        return redirect(url_for("categories.index"))

    parent = None

    if parent_id:
        parent = Category.query.filter_by(
            id=parent_id,
            user_id=current_user.id
        ).first()

        if parent is None:
            flash("Catégorie parente invalide.", "danger")
            return redirect(url_for("categories.index"))

    category = Category(
        name=name,
        parent_id=parent.id if parent else None,
        user_id=current_user.id
    )

    db.session.add(category)
    db.session.commit()

    flash("Catégorie créée avec succès.", "success")
    return redirect(url_for("categories.index"))
