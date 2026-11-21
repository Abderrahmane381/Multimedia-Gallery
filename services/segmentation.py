
import cv2
import numpy as np


def automatic_segmentation(image_path):
    """
    Segmentation automatique avec GrabCut.
    Le rectangle initial couvre presque toute l'image.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Impossible de lire l'image.")

    height, width = image.shape[:2]

    if width < 3 or height < 3:
        raise ValueError("Image trop petite pour la segmentation.")

    # Rectangle initial avec une marge de 5 %
    margin_x = max(1, int(width * 0.05))
    margin_y = max(1, int(height * 0.05))

    rect = (
        margin_x,
        margin_y,
        width - 2 * margin_x,
        height - 2 * margin_y
    )

    mask = np.zeros((height, width), np.uint8)

    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        image,
        mask,
        rect,
        background_model,
        foreground_model,
        5,
        cv2.GC_INIT_WITH_RECT
    )

    # Pixels appartenant au premier plan
    foreground_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    return image, foreground_mask


def manual_segmentation(image_path, x, y, width, height):
    """
    Segmentation assistée par l'utilisateur.
    L'utilisateur indique un rectangle autour de l'objet.
    GrabCut affine ensuite la sélection.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Impossible de lire l'image.")

    image_height, image_width = image.shape[:2]

    x = int(x)
    y = int(y)
    width = int(width)
    height = int(height)

    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("Coordonnées invalides.")

    if x + width > image_width or y + height > image_height:
        raise ValueError("Le rectangle dépasse les limites de l'image.")

    if width < 3 or height < 3:
        raise ValueError("Le rectangle est trop petit.")

    mask = np.zeros((image_height, image_width), np.uint8)

    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        image,
        mask,
        (x, y, width, height),
        background_model,
        foreground_model,
        5,
        cv2.GC_INIT_WITH_RECT
    )

    foreground_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    return image, foreground_mask


def save_segmentation(image, mask, output_path):
    """
    Enregistre l'objet avec un fond transparent.
    """

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    image_rgb[:, :, 3] = mask

    success = cv2.imwrite(output_path, image_rgb)

    if not success:
        raise ValueError("Impossible d'enregistrer le résultat.")

    return output_path


def save_background(image, mask, output_path):
    """
    Enregistre séparément le background.
    """

    background_mask = cv2.bitwise_not(mask)

    background = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    background[:, :, 3] = background_mask

    success = cv2.imwrite(output_path, background)

    if not success:
        raise ValueError("Impossible d'enregistrer le background.")

    return output_path
