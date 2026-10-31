"""Faithful threshold operations from TP2 preprocessing notebook."""
from time import perf_counter

import cv2
import numpy as np


def _gray(image):
    arr = np.asarray(image)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("TP2 attend une image 2D non vide")
    return np.clip(arr, 0, 255).astype(np.uint8)


def manual_threshold_tp2(image, threshold=50):
    """TP2 cell 10: cv2.THRESH_BINARY; default notebook threshold is 50."""
    start = perf_counter()
    _, binary = cv2.threshold(_gray(image), int(threshold), 255, cv2.THRESH_BINARY)
    return binary, perf_counter() - start


def otsu_tp2(image):
    """TP2 cell 12: automatic Otsu with bright-class binary interpretation."""
    start = perf_counter()
    threshold, binary = cv2.threshold(
        _gray(image), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary, perf_counter() - start, float(threshold)
