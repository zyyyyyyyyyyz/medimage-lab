from pathlib import Path

import cv2
import numpy as np
from skimage.filters import gaussian, median
from skimage.morphology import disk

from core.algorithms import ensure_gray, preprocess, segmentation
from core.io_utils import load_rgb
from core.tp3 import dice_tp3, iou_tp3, prepare_tp3_reference, segment_tp3
from core.tp2 import manual_threshold_tp2, otsu_tp2


ROOT = Path(__file__).resolve().parents[1]


def test_tp2_filter_numerical_parity():
    gray = ensure_gray(load_rgb(ROOT / "assets" / "bagls_test_2.png"))
    expected_median = median(gray, footprint=disk(3)).astype(np.uint8)
    expected_gaussian = np.clip(
        gaussian(gray, sigma=1.0, preserve_range=True), 0, 255
    ).astype(np.uint8)
    np.testing.assert_array_equal(preprocess(gray, "Médian"), expected_median)
    np.testing.assert_array_equal(preprocess(gray, "Gaussien"), expected_gaussian)


def test_tp2_threshold_numerical_parity():
    gray = ensure_gray(load_rgb(ROOT / "assets" / "bagls_test_2.png"))
    _, expected_manual = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    expected_otsu_threshold, expected_otsu = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    actual_manual, _ = manual_threshold_tp2(gray)
    actual_otsu, _, actual_otsu_threshold = otsu_tp2(gray)
    np.testing.assert_array_equal(actual_manual, expected_manual)
    np.testing.assert_array_equal(actual_otsu, expected_otsu)
    assert actual_otsu_threshold == expected_otsu_threshold


def test_tp3_recorded_bagls_regression():
    rgb = load_rgb(ROOT / "assets" / "bagls_test_2.png")
    reference = ensure_gray(load_rgb(ROOT / "assets" / "bagls_test_2_mask.png"))
    expected = {
        "Mean Shift": (0.8337, 0.7148),
        "K-means": (0.8301, 0.7095),
    }
    for method, (expected_dice, expected_iou) in expected.items():
        working, mask, _ = segment_tp3(rgb, method)
        resized_reference = prepare_tp3_reference(
            reference, (rgb.shape[1], rgb.shape[0]), mask.shape
        )
        assert working.shape == (256, 114, 3)
        assert round(dice_tp3(mask, resized_reference), 4) == expected_dice
        assert round(iou_tp3(mask, resized_reference), 4) == expected_iou


def test_tp3_chan_vese_is_deterministic_and_binary():
    rgb = load_rgb(ROOT / "assets" / "bagls_test_2.png")
    _, first, _ = segment_tp3(rgb, "Chan–Vese")
    _, second, _ = segment_tp3(rgb, "Chan–Vese")
    np.testing.assert_array_equal(first, second)
    assert first.shape == (256, 114)
    assert set(np.unique(first)).issubset({0, 255})
