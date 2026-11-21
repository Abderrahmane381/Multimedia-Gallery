
import os
import cv2
import numpy as np

from services.color_features import extract_color_features
from services.texture_features import extract_texture_features


def extract_all_features(image_path, mode="global", object_mask_path=None):
    """
    Extrait les descripteurs couleur et texture.

    Modes :
        global     : image complète
        object     : objet segmenté
        background : arrière-plan

    Retourne un dictionnaire contenant les descripteurs.
    """

    if mode not in ("global", "object", "background"):
        raise ValueError("Mode invalide.")

    if not os.path.isfile(image_path):
        raise FileNotFoundError("Image introuvable.")

    mask_path = None

    # Préparer le masque selon le mode demandé
    if mode in ("object", "background"):

        if not object_mask_path:
            raise ValueError("Le chemin du masque est obligatoire.")

        segmented = cv2.imread(
            object_mask_path,
            cv2.IMREAD_UNCHANGED
        )

        if segmented is None:
            raise ValueError("Impossible de lire le masque.")

        if segmented.ndim == 3 and segmented.shape[2] == 4:
            object_mask = segmented[:, :, 3] > 0
        elif segmented.ndim == 2:
            object_mask = segmented > 0
        else:
            raise ValueError("Format du masque invalide.")

        if mode == "object":
            selected_mask = object_mask
        else:
            selected_mask = ~object_mask

        # Construire un masque PNG avec transparence
        mask_path = os.path.splitext(image_path)[0] + f"_{mode}_mask.png"

        mask_image = np.zeros(
            (*selected_mask.shape, 4),
            dtype=np.uint8
        )

        mask_image[:, :, 3] = selected_mask.astype(np.uint8) * 255

        if not cv2.imwrite(mask_path, mask_image):
            raise ValueError("Impossible de créer le masque temporaire.")

    # Extraire les descripteurs de couleur
    color_features = extract_color_features(
        image_path,
        mask_path=mask_path
    )

    # Extraire les descripteurs de texture
    texture_features = extract_texture_features(
        image_path,
        mask_path=mask_path
    )

    # Supprimer le masque temporaire
    if mask_path is not None and os.path.exists(mask_path):
        os.remove(mask_path)

    return {
        "mode": mode,
        "color": color_features,
        "texture": texture_features
    }
