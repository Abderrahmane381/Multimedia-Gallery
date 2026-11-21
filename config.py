
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = "multimedia-gallery-secret-key"
    SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(BASE_DIR, "database", "gallery.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    GENERATED_FOLDER = os.path.join(BASE_DIR, "generated")

    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB
    FEATURE_INDEX_PATH = os.path.join(
        BASE_DIR,
        "database",
        "ghim10k_global_features.npz"
    )
