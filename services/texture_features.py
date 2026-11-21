
import cv2
import numpy as np

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


def masked_glcm(gray_image, mask, distances, angles, levels=32):
    """
    Calcule une GLCM en ignorant les paires de pixels
    dont l'un des deux pixels est hors de l'objet.
    """

    height, width = gray_image.shape

    glcm = np.zeros(
        (levels, levels, len(distances), len(angles)),
        dtype=np.float64
    )

    for d_index, distance in enumerate(distances):
        for a_index, angle in enumerate(angles):

            # Décalage correspondant à la distance et à l'angle
            dx = int(round(distance * np.cos(angle)))
            dy = int(round(-distance * np.sin(angle)))

            # Éviter un décalage nul
            if dx == 0 and dy == 0:
                dx = 1

            # Définir les deux zones comparées
            y1_start = max(0, -dy)
            y1_end = min(height, height - dy)
            x1_start = max(0, -dx)
            x1_end = min(width, width - dx)

            y2_start = y1_start + dy
            y2_end = y1_end + dy
            x2_start = x1_start + dx
            x2_end = x1_end + dx

            first = gray_image[
                y1_start:y1_end,
                x1_start:x1_end
            ]

            second = gray_image[
                y2_start:y2_end,
                x2_start:x2_end
            ]

            # Une paire est valide seulement si les deux pixels
            # appartiennent à l'objet
            valid = (
                mask[y1_start:y1_end, x1_start:x1_end]
                & mask[y2_start:y2_end, x2_start:x2_end]
            )

            values1 = first[valid]
            values2 = second[valid]

            if len(values1) == 0:
                continue

            # Ajouter les paires dans les deux sens
            np.add.at(glcm[:, :, d_index, a_index],
                      (values1, values2), 1)

            np.add.at(glcm[:, :, d_index, a_index],
                      (values2, values1), 1)

            # Normaliser la matrice
            total = glcm[:, :, d_index, a_index].sum()

            if total > 0:
                glcm[:, :, d_index, a_index] /= total

    return glcm


def extract_texture_features(image_path, mask_path=None):
    """
    Extrait les descripteurs de texture.

    Sans mask_path :
        Analyse toute l'image.

    Avec mask_path :
        Analyse uniquement l'objet segmenté.

    Descripteurs :
        - GLCM : contraste, corrélation, énergie, homogénéité
        - LBP : histogramme des motifs binaires locaux
    """

    # 1. Charger l'image en niveaux de gris
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Impossible de lire l'image.")

    # 2. Construire le masque
    if mask_path is not None:

        segmented = cv2.imread(
            mask_path,
            cv2.IMREAD_UNCHANGED
        )

        if segmented is None:
            raise ValueError("Impossible de lire le masque.")

        if segmented.shape[:2] != image.shape:
            raise ValueError("L'image et le masque doivent avoir les mêmes dimensions.")

        if segmented.ndim == 3 and segmented.shape[2] == 4:
            mask = segmented[:, :, 3] > 0

        elif segmented.ndim == 2:
            mask = segmented > 0

        else:
            raise ValueError("Le masque doit contenir un canal alpha ou être en niveaux de gris.")

    else:
        mask = np.ones(image.shape, dtype=bool)

    if not np.any(mask):
        raise ValueError("Le masque ne contient aucun pixel d'objet.")

    # 3. Quantification en 32 niveaux de gris
    image_glcm = (image // 8).astype(np.uint8)

    distances = [1, 2]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    # 4. Calcul GLCM
    if mask_path is None:

        glcm = graycomatrix(
            image_glcm,
            distances=distances,
            angles=angles,
            levels=32,
            symmetric=True,
            normed=True
        )

    else:

        glcm = masked_glcm(
            image_glcm,
            mask,
            distances,
            angles,
            levels=32
        )

    # 5. Propriétés GLCM
    glcm_features = {}

    for prop in [
        "contrast",
        "correlation",
        "energy",
        "homogeneity"
    ]:

        values = graycoprops(glcm, prop)

        glcm_features[prop] = {
            "mean": float(np.mean(values)),
            "values": values.tolist()
        }

    # 6. Calcul LBP
    radius = 3
    points = 8 * radius

    lbp = local_binary_pattern(
        image,
        P=points,
        R=radius,
        method="uniform"
    )

    # 7. Écarter les pixels proches du bord du masque.
    # Leur voisinage LBP pourrait contenir du background.
    if mask_path is not None:

        kernel_size = 2 * radius + 1
        kernel = np.ones(
            (kernel_size, kernel_size),
            dtype=np.uint8
        )

        eroded_mask = cv2.erode(
            mask.astype(np.uint8),
            kernel,
            iterations=1
        ).astype(bool)

        # Si l'objet est trop petit, utiliser le masque original
        if np.count_nonzero(eroded_mask) > 0:
            lbp_mask = eroded_mask
        else:
            lbp_mask = mask

    else:
        lbp_mask = mask

    lbp_values = lbp[lbp_mask]

    # 8. Histogramme LBP normalisé
    number_of_bins = points + 2

    lbp_histogram, _ = np.histogram(
        lbp_values,
        bins=np.arange(0, number_of_bins + 1),
        range=(0, number_of_bins)
    )

    lbp_histogram = lbp_histogram.astype(np.float64)
    lbp_histogram /= (lbp_histogram.sum() + 1e-10)

    return {
        "segmentation_used": mask_path is not None,
        "analyzed_pixels": int(np.count_nonzero(mask)),
        "lbp_analyzed_pixels": int(np.count_nonzero(lbp_mask)),
        "glcm": glcm_features,
        "lbp": {
            "radius": radius,
            "points": points,
            "histogram": lbp_histogram.tolist()
        }
    }
