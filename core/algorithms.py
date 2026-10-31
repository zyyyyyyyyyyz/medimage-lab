"""Shared display helpers and explicitly labelled web extensions.

Faithful TP2 thresholds are in ``core.tp2`` and faithful RGB TP3 segmentation
is in ``core.tp3``. Nothing in this module should be presented as TP3 parity.
Non destiné au diagnostic.
"""
from time import perf_counter
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.segmentation import watershed
from skimage.filters import sobel
from skimage.measure import label as connected_labels
from skimage.filters import gaussian as sk_gaussian, median as sk_median
from skimage.morphology import disk


def _gray_uint8(image, name="Image"):
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError(f"{name} doit être une image 2D en niveaux de gris")
    if arr.size == 0:
        raise ValueError(f"{name} vide")
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{name} doit contenir des valeurs numériques")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contient des valeurs non finies")
    return np.clip(arr, 0, 255).astype(np.uint8)


def ensure_gray(rgb):
    arr = np.asarray(rgb)
    if arr.ndim == 2:
        return _gray_uint8(arr)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError("Image attendue au format gris, RGB ou RGBA")
    return cv2.cvtColor(np.clip(arr[:, :, :3], 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)


def preprocess(gray, method="Aucun", intensity=3):
    gray = _gray_uint8(gray)
    smallest = min(gray.shape)
    requested = max(3, int(intensity) | 1)
    max_odd = smallest if smallest % 2 else smallest - 1
    k = min(requested, max_odd)
    if method == "Aucun": return gray.copy()
    if k < 3 and method in ("Médian", "Gaussien"):
        return gray.copy()
    # TP2 cells 5/7: disk(3), skimage median and gaussian sigma=1.
    if method == "Médian": return sk_median(gray, footprint=disk(3)).astype(np.uint8)
    if method == "Gaussien":
        return np.clip(sk_gaussian(gray, sigma=1.0, preserve_range=True), 0, 255).astype(np.uint8)
    if method == "Bilatéral": return cv2.bilateralFilter(gray, 7, 45, 45)
    if method == "CLAHE": return cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8)).apply(gray)
    if method == "Égalisation histogramme": return cv2.equalizeHist(gray)
    raise ValueError(f"Prétraitement inconnu : {method}")


def clean_mask(mask, min_area=0, fill=True):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("Le masque doit être une image 2D non vide")
    if fill: mask = binary_fill_holes(mask)
    if min_area > 0:
        labels = connected_labels(mask, connectivity=2)
        sizes = np.bincount(labels.ravel())
        if len(sizes): sizes[0] = 0
        mask = (labels > 0) & (sizes[labels] >= min_area)
    return (mask.astype(np.uint8) * 255)


def segmentation(gray, method, foreground="Sombre", clusters=3, threshold=127):
    """Web extensions only. TP2 thresholds live in tp2; TP3 methods live in tp3."""
    gray = _gray_uint8(gray)
    if foreground not in ("Sombre", "Claire"):
        raise ValueError("La région doit être 'Sombre' ou 'Claire'")
    h, w = gray.shape
    # Le TP3 d'origine travaillait à une résolution réduite pour Mean Shift.
    max_side = 512
    factor = min(1.0, max_side / max(h, w))
    small = cv2.resize(gray, (max(1, round(w * factor)), max(1, round(h * factor))),
                       interpolation=cv2.INTER_AREA) if factor < 1 else gray
    dark = foreground == "Sombre"
    if np.max(small) == np.min(small):
        return np.zeros_like(gray)
    if method == "Otsu":
        _, mask = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if dark: mask = 255 - mask
    elif method == "Seuil manuel":
        threshold = int(np.clip(threshold, 0, 255))
        mask = (small <= threshold if dark else small >= threshold).astype(np.uint8) * 255
    elif method == "Watershed":
        # Intensity-based markers, pedagogical (not automatic anatomical delineation).
        p25, p75 = np.percentile(small, [25, 75])
        markers = np.zeros(small.shape, dtype=np.int32)
        markers[small <= p25] = 1
        markers[small >= p75] = 2
        reg = watershed(sobel(small.astype(np.float32)/255.), markers=markers)
        mask = ((reg == (1 if dark else 2)).astype(np.uint8) * 255)
    else: raise ValueError(f"Méthode inconnue : {method}")
    if mask.shape != gray.shape:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.uint8) * 255


def scores(prediction, reference):
    a = np.asarray(prediction) > 127
    b = np.asarray(reference) > 127
    if a.ndim != 2 or b.ndim != 2: raise ValueError("Les masques doivent être des images 2D")
    if a.shape != b.shape: raise ValueError("Les masques doivent avoir les mêmes dimensions")
    intersection = np.logical_and(a, b).sum()
    count = a.sum() + b.sum()
    union = np.logical_or(a, b).sum()
    return {"Dice": float(2 * intersection / count) if count else 1.0,
            "IoU": float(intersection / union) if union else 1.0}


def overlay(rgb, mask, alpha=0.45):
    image = np.asarray(rgb, dtype=np.uint8).copy()
    on = np.asarray(mask) > 127
    if on.shape != image.shape[:2]: raise ValueError("Masque et image de tailles différentes")
    image[on] = ((1 - alpha) * image[on] + alpha * np.array([0, 230, 150])).astype(np.uint8)
    boundary = cv2.morphologyEx((on.astype(np.uint8)*255), cv2.MORPH_GRADIENT,
                                np.ones((3, 3), np.uint8)) > 0
    image[boundary] = [255, 80, 80]
    return image


def timed_segmentation(gray, method, **kwargs):
    t0 = perf_counter()
    result = segmentation(gray, method, **kwargs)
    return result, perf_counter() - t0
