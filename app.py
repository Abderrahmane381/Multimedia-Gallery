
import os
from flask import Flask, app, redirect, url_for
from flask_login import login_required, current_user
from api import init_api
from config import Config
from extensions import db, login_manager



def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["GENERATED_FOLDER"], exist_ok=True)

    from models import User, Image, Category
    from routes.auth import auth
    from routes.gallery import gallery
    from routes.categories import categories
    from routes.transformations import transformations
    from routes.search import search
    from routes.descriptors import descriptors

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))

    app.register_blueprint(auth)
    app.register_blueprint(gallery)
    app.register_blueprint(categories)
    app.register_blueprint(transformations)
    app.register_blueprint(search)
    app.register_blueprint(descriptors)

    # Initialiser l'API REST
    init_api(app)


    @app.route("/")
    @login_required
    def home():
        return redirect(url_for("gallery.show_gallery"))

    with app.app_context():
        db.create_all()

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
