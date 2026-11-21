
from extensions import db


class Category(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    parent_id = db.Column(db.Integer, db.ForeignKey("category.id"))
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True)

    children = db.relationship(
        "Category",
        backref=db.backref("parent", remote_side=[id]),
        lazy=True
    )

    user = db.relationship("User", backref="categories")
