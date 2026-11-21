
import cv2
import numpy as np


def extract_color_features(image_path, mask_path=None, bins=8, k=5):
    """
    Extrait les descripteurs de couleur.

    Si mask_path est fourni :
    seuls les pixels non transparents du PNG segmenté
    sont utilisés.

    Descripteurs :
    - Histogrammes HSV
    - Moments HSV : moyenne, écart-type, asymétrie
    - Couleurs dominantes RGB avec K-Means
    """

    # 1. Charger l'image originale
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Impossible de lire l'image originale.")

    # 2. Créer le masque des pixels à analyser
    if mask_path is not None:
        segmented = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if segmented is None:
            raise ValueError("Impossible de lire le masque de segmentation.")

        if segmented.shape[:2] != image.shape[:2]:
            raise ValueError("L'image et le masque doivent avoir les mêmes dimensions.")

        # Le PNG segmenté contient un canal alpha
        if segmented.ndim == 3 and segmented.shape[2] == 4:
            mask = segmented[:, :, 3] > 0
        elif segmented.ndim == 2:
            mask = segmented > 0
        else:
            raise ValueError("Le masque doit être un PNG avec transparence ou une image en niveaux de gris.")

    else:
        # Sans masque : analyser toute l'image
        mask = np.ones(image.shape[:2], dtype=bool)

    if not np.any(mask):
        raise ValueError("Aucun pixel d'objet détecté dans le masque.")

    # 3. Convertir en HSV et RGB
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Garder uniquement les pixels appartenant à l'objet
    hsv_pixels = hsv[mask]
    rgb_pixels = rgb[mask]

    # 4. Histogrammes HSV
    histograms = {}

    channels = {
        "H": (0, 180),
        "S": (0, 256),
        "V": (0, 256)
    }

    for index, (channel_name, value_range) in enumerate(channels.items()):
        hist, _ = np.histogram(
            hsv_pixels[:, index],
            bins=bins,
            range=value_range
        )

        # Normalisation en distribution de probabilités
        hist = hist.astype(np.float64)
        hist /= hist.sum()

        histograms[channel_name] = hist.tolist()

    # 5. Moments statistiques HSV
    moments = {}

    for index, channel_name in enumerate(["H", "S", "V"]):
        channel = hsv_pixels[:, index].astype(np.float64)

        mean = float(np.mean(channel))
        std = float(np.std(channel))

        if std > 0:
            skewness = float(np.mean(((channel - mean) / std) ** 3))
        else:
            skewness = 0.0

        moments[channel_name] = {
            "mean": mean,
            "std": std,
            "skewness": skewness
        }

    # 6. Couleurs dominantes RGB avec K-Means
    pixels = rgb_pixels.astype(np.float32)

    # Échantillonnage pour accélérer le calcul
    if len(pixels) > 10000:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(pixels), 10000, replace=False)
        sample = pixels[indices]
    else:
        sample = pixels

    number_of_colors = min(k, len(sample))

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.2
    )

    _, labels, centers = cv2.kmeans(
        sample,
        number_of_colors,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS
    )

    counts = np.bincount(
        labels.flatten(),
        minlength=number_of_colors
    )

    proportions = counts / counts.sum()

    dominant_colors = []

    for center, proportion in zip(centers, proportions):
        dominant_colors.append({
            "rgb": [int(value) for value in center],
            "percentage": round(float(proportion) * 100, 2)
        })

    dominant_colors.sort(
        key=lambda color: color["percentage"],
        reverse=True
    )

    return {
        "segmentation_used": mask_path is not None,
        "analyzed_pixels": int(np.count_nonzero(mask)),
        "histograms": histograms,
        "moments": moments,
        "dominant_colors": dominant_colors
    }
