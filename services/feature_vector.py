
import numpy as np

from services.feature_extraction import extract_all_features


def _flatten_numeric(data):
    """
    Transforme une structure composée de listes, dictionnaires
    et nombres en une liste de valeurs numériques.
    """
    values = []

    if isinstance(data, dict):
        for key in sorted(data.keys()):
            values.extend(_flatten_numeric(data[key]))

    elif isinstance(data, (list, tuple, np.ndarray)):
        for item in data:
            values.extend(_flatten_numeric(item))

    elif isinstance(data, (int, float, np.integer, np.floating)):
        value = float(data)

        if np.isfinite(value):
            values.append(value)

    return values


def _get_histograms(color):
    """Récupère les histogrammes HSV dans l'ordre H, S, V."""
    histograms = color.get("histograms", {})

    if not isinstance(histograms, dict):
        raise ValueError("Format des histogrammes couleur invalide.")

    possible_keys = [
        ("H", "h", "hue"),
        ("S", "s", "saturation"),
        ("V", "v", "value")
    ]

    result = []

    for aliases in possible_keys:
        histogram = None

        for key in aliases:
            if key in histograms:
                histogram = histograms[key]
                break

        if histogram is None:
            raise ValueError(
                f"Histogramme HSV manquant : {aliases[0]}"
            )

        result.extend(_flatten_numeric(histogram))

    return result


def _get_moments(color):
    """Récupère les moments statistiques des canaux HSV."""
    moments = color.get("moments", {})

    if not isinstance(moments, dict):
        raise ValueError("Format des moments couleur invalide.")

    result = []

    for channel in ("H", "S", "V"):
        channel_data = None

        for key in (channel, channel.lower(),
                    {"H": "hue", "S": "saturation", "V": "value"}[channel]):
            if key in moments:
                channel_data = moments[key]
                break

        if channel_data is None:
            raise ValueError(f"Moments du canal {channel} manquants.")

        # Ordre fixe : moyenne, écart-type, asymétrie
        if isinstance(channel_data, dict):
            for key in ("mean", "std", "skewness"):
                if key not in channel_data:
                    raise ValueError(
                        f"Moment {key} manquant pour le canal {channel}."
                    )
                result.extend(_flatten_numeric(channel_data[key]))
        else:
            result.extend(_flatten_numeric(channel_data))

    return result


def _get_dominant_colors(color):
    """Encode les 5 couleurs dominantes en RGB normalisé + pourcentage."""
    dominant = color.get("dominant_colors", [])

    if not isinstance(dominant, list) or len(dominant) != 5:
        raise ValueError(
            "Il faut exactement 5 couleurs dominantes."
        )

    result = []

    for item in dominant:
        rgb = item.get("rgb")
        percentage = item.get("percentage")

        if rgb is None or len(rgb) != 3 or percentage is None:
            raise ValueError("Format d'une couleur dominante invalide.")

        # RGB entre 0 et 1, pourcentage entre 0 et 1
        result.extend([float(value) / 255.0 for value in rgb])
        result.append(float(percentage) / 100.0)

    return result


def _get_glcm(texture):
    """Récupère les moyennes des propriétés GLCM."""
    glcm = texture.get("glcm", {})

    properties = (
        "contrast",
        "correlation",
        "energy",
        "homogeneity"
    )

    result = []

    for prop in properties:
        if prop not in glcm or "mean" not in glcm[prop]:
            raise ValueError(f"Propriété GLCM manquante : {prop}")

        result.append(float(glcm[prop]["mean"]))

    return result


def _get_lbp(texture):
    """Récupère l'histogramme LBP."""
    lbp = texture.get("lbp")

    if lbp is None:
        raise ValueError("Descripteur LBP manquant.")

    # Accepte aussi un dictionnaire contenant l'histogramme
    if isinstance(lbp, dict):
        for key in ("histogram", "hist", "values"):
            if key in lbp:
                lbp = lbp[key]
                break

    result = _flatten_numeric(lbp)

    if len(result) != 26:
        raise ValueError(
            f"Histogramme LBP attendu : 26 valeurs, obtenu : {len(result)}."
        )

    return result


def build_feature_vector(features):
    """
    Construit un vecteur numérique à partir des descripteurs
    couleur et texture produits par extract_all_features().

    Composition :
    - 24 valeurs : histogrammes HSV
    - 9 valeurs : moments HSV
    - 20 valeurs : 5 couleurs dominantes (RGB + pourcentage)
    - 4 valeurs : propriétés GLCM
    - 26 valeurs : histogramme LBP

    Total attendu : 83 valeurs.
    """
    if not isinstance(features, dict):
        raise ValueError("Les caractéristiques doivent être un dictionnaire.")

    color = features.get("color")
    texture = features.get("texture")

    if not isinstance(color, dict) or not isinstance(texture, dict):
        raise ValueError("Descripteurs couleur ou texture manquants.")

    vector = []

    vector.extend(_get_histograms(color))
    vector.extend(_get_moments(color))
    vector.extend(_get_dominant_colors(color))
    vector.extend(_get_glcm(texture))
    vector.extend(_get_lbp(texture))

    vector = np.asarray(vector, dtype=np.float32)

    if vector.shape[0] != 83:
        raise ValueError(
            f"Vecteur attendu : 83 dimensions, obtenu : {vector.shape[0]}."
        )

    if not np.all(np.isfinite(vector)):
        raise ValueError("Le vecteur contient des valeurs invalides.")

    return vector


def extract_feature_vector(
    image_path,
    mode="global",
    object_mask_path=None
):
    """
    Extrait les descripteurs puis construit le vecteur correspondant.

    Modes disponibles : global, object, background.
    """
    features = extract_all_features(
        image_path=image_path,
        mode=mode,
        object_mask_path=object_mask_path
    )

    return build_feature_vector(features)


def feature_vector_to_list(vector):
    """Convertit un vecteur NumPy en liste Python."""
    return np.asarray(vector, dtype=np.float32).tolist()
