
import os
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
ZIP_NAME = "Multimedia_Gallery_Final.zip"

EXCLUDED_DIRS = {
    "venv",
    "dataset",
    "__pycache__",
    "generated",
    "uploads",
    ".git"
}

EXCLUDED_FILES = {
    ZIP_NAME,
    "create_zip.py"
}

with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as archive:
    for current_root, dirs, files in os.walk(ROOT):
        dirs[:] = [
            d for d in dirs
            if d not in EXCLUDED_DIRS
        ]

        for filename in files:
            if filename in EXCLUDED_FILES:
                continue

            full_path = os.path.join(current_root, filename)
            relative_path = os.path.relpath(full_path, ROOT)

            archive.write(full_path, relative_path)

print(f"Archive créée : {ZIP_NAME}")
