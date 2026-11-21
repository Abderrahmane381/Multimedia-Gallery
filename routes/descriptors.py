
from flask import Blueprint, render_template, request
from flask_login import login_required, current_user

from models.image import Image
from services.feature_extraction import extract_all_features
from services.segmentation import automatic_segmentation, save_segmentation

import os
import tempfile

descriptors = Blueprint("descriptors", __name__)


@descriptors.route("/image/<int:image_id>/descriptors")
@login_required
def image_descriptors(image_id):

    image = Image.query.filter_by(
        id=image_id,
        user_id=current_user.id
    ).first_or_404()

    mode = request.args.get("mode", "global")

    if mode not in ("global", "object", "background"):
        mode = "global"

    try:
        if mode == "global":
            features = extract_all_features(
                image.filepath,
                mode="global"
            )
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                segmented_image, mask = automatic_segmentation(
                    image.filepath
                )

                mask_path = os.path.join(temp_dir, "mask.png")

                save_segmentation(
                    segmented_image,
                    mask,
                    mask_path
                )

                features = extract_all_features(
                    image.filepath,
                    mode=mode,
                    object_mask_path=mask_path
                )

        return render_template(
            "descriptors.html",
            image=image,
            features=features,
            mode=mode
        )

    except Exception as error:
        return f"Erreur de segmentation ou extraction : {error}", 500
