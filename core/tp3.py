"""Faithful, UI-independent extraction of executed TP3 BAGLS operations.

Source: 03-Segmentation_Mme_ATTIA.ipynb, cells 9, 14, 18, 20 and 24.
The implementation keeps the notebook's RGB clustering, 256-pixel thumbnail,
fixed random states, dark-class interpretation and largest-component cleanup.
"""
from time import perf_counter

import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes, median_filter
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from skimage.color import rgb2gray
from skimage.measure import label
from skimage.segmentation import chan_vese


TP3_METHODS = ("Mean Shift", "K-means", "Chan–Vese")


def prepare_tp3_image(rgb, max_size=256):
    """Equivalent to Image.fromarray(...).convert('RGB').thumbnail((256,256))."""
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4) or arr.size == 0:
        raise ValueError("TP3 attend une image RGB non vide")
    image_pil = Image.fromarray(np.clip(arr[:, :, :3], 0, 255).astype(np.uint8)).convert("RGB")
    image_pil.thumbnail((max_size, max_size))
    return np.array(image_pil)


def _dark_class(labels, grayscale):
    numbers = np.unique(labels)
    means = [grayscale[labels == number].mean() for number in numbers]
    return numbers[np.argmin(means)]


def clean_tp3_mask(mask):
    """Notebook nettoyer(): retain largest connected component, then fill holes."""
    regions = label(np.asarray(mask, dtype=bool))
    sizes = np.bincount(regions.ravel())
    sizes[0] = 0
    if sizes.max() == 0:
        return np.zeros_like(mask, dtype=bool)
    return binary_fill_holes(regions == np.argmax(sizes))


def segment_tp3(rgb, method):
    """Return (working RGB, binary uint8 mask, seconds) at TP3 resolution."""
    if method not in TP3_METHODS:
        raise ValueError(f"Méthode TP3 inconnue : {method}")
    image = prepare_tp3_image(rgb)
    height, width = image.shape[:2]
    grayscale = rgb2gray(image)
    start = perf_counter()

    if method == "Mean Shift":
        filtered = median_filter(image, size=(3, 3, 1))
        pixels = filtered.reshape(-1, 3).astype(float)
        bandwidth = estimate_bandwidth(
            pixels, quantile=0.2, n_samples=min(1000, len(pixels)), random_state=0
        )
        model = MeanShift(bandwidth=max(bandwidth, 1.0), bin_seeding=True)
        labels = model.fit_predict(pixels).reshape(height, width)
        selected = _dark_class(labels, grayscale)
        mask = clean_tp3_mask(labels == selected)
    elif method == "K-means":
        pixels = image.reshape(-1, 3).astype(float)
        model = KMeans(n_clusters=3, random_state=0, n_init=10)
        labels = model.fit_predict(pixels).reshape(height, width)
        selected = _dark_class(labels, grayscale)
        mask = clean_tp3_mask(labels == selected)
    else:
        regions = chan_vese(
            grayscale, mu=0.25, lambda1=1, lambda2=1, tol=0.001,
            max_num_iter=200, dt=0.5, init_level_set="checkerboard",
        )
        selected = _dark_class(regions.astype(int), grayscale)
        mask = clean_tp3_mask(regions == selected)

    return image, mask.astype(np.uint8) * 255, perf_counter() - start


def prepare_tp3_reference(reference_gray, original_size, working_shape):
    """Reproduce TP3 reference validation and nearest-neighbor resize."""
    reference = np.asarray(reference_gray)
    if reference.ndim != 2:
        raise ValueError("Le masque TP3 doit être en niveaux de gris")
    expected_width, expected_height = original_size
    if reference.shape != (expected_height, expected_width):
        raise ValueError("Le masque doit avoir les dimensions de l'image originale.")
    height, width = working_shape
    resized = Image.fromarray(np.clip(reference, 0, 255).astype(np.uint8)).resize(
        (width, height), Image.Resampling.NEAREST
    )
    return (np.array(resized) > 0).astype(np.uint8) * 255


def dice_tp3(a, b):
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("Les masques doivent avoir la même taille.")
    intersection, total = np.logical_and(a, b).sum(), a.sum() + b.sum()
    return float(2 * intersection / total) if total else 1.0


def iou_tp3(a, b):
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("Les masques doivent avoir la même taille.")
    intersection, union = np.logical_and(a, b).sum(), np.logical_or(a, b).sum()
    return float(intersection / union) if union else 1.0
