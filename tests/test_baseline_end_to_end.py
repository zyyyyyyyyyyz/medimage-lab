from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from core.algorithms import clean_mask, ensure_gray, preprocess, scores, segmentation
from core.io_utils import load_rgb, mask_png_bytes, results_csv_bytes
from core.tp3 import TP3_METHODS, prepare_tp3_reference, segment_tp3
from core.tp2 import manual_threshold_tp2, otsu_tp2


ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = ROOT / "assets" / "bagls_test_2.png"
MASK_PATH = ROOT / "assets" / "bagls_test_2_mask.png"


def test_upload_validation_and_all_preprocessors():
    rgb = load_rgb(IMAGE_PATH)
    gray = ensure_gray(rgb)
    assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
    for method in ("Aucun", "Médian", "Gaussien", "Bilatéral", "CLAHE", "Égalisation histogramme"):
        result = preprocess(gray, method, 5)
        assert result.shape == gray.shape
        assert result.dtype == np.uint8
        assert np.isfinite(result).all()
    with pytest.raises(ValueError, match="illisible|format"):
        load_rgb(b"not an image")
    with pytest.raises(ValueError, match="trop grande"):
        load_rgb(IMAGE_PATH, max_pixels=100)


def test_bagls_all_conventional_methods_and_exports():
    gray = ensure_gray(load_rgb(IMAGE_PATH))
    reference = ensure_gray(load_rgb(MASK_PATH))
    assert gray.shape == reference.shape

    records = []
    for method in ("Seuil manuel", "Otsu", "K-means", "Mean Shift", "Chan–Vese", "Watershed"):
        if method in TP3_METHODS:
            _, mask, _ = segment_tp3(load_rgb(IMAGE_PATH), method)
            metric_reference = prepare_tp3_reference(
                reference, (gray.shape[1], gray.shape[0]), mask.shape
            )
        elif method == "Seuil manuel":
            mask, _ = manual_threshold_tp2(gray, 50)
            metric_reference = reference
        elif method == "Otsu":
            mask, _, _ = otsu_tp2(gray)
            metric_reference = reference
        else:
            mask = segmentation(gray, method, foreground="Sombre", clusters=3, threshold=127)
            mask = clean_mask(mask, min_area=0, fill=False)
            metric_reference = reference
        assert mask.shape == metric_reference.shape
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 255})
        metric = scores(mask, metric_reference)
        assert 0.0 <= metric["Dice"] <= 1.0
        assert 0.0 <= metric["IoU"] <= 1.0
        records.append({"Méthode": method, **metric})

        png = mask_png_bytes(mask)
        exported = np.asarray(Image.open(BytesIO(png)))
        assert exported.shape == mask.shape
        assert set(np.unique(exported)).issubset({0, 255})

    csv_payload = results_csv_bytes(records)
    table = pd.read_csv(BytesIO(csv_payload))
    assert list(table.columns) == ["Méthode", "Dice", "IoU"]
    assert len(table) == 6


def test_misaligned_ground_truth_is_rejected_by_metrics():
    prediction = np.zeros((20, 20), dtype=np.uint8)
    reference = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError, match="mêmes dimensions"):
        scores(prediction, reference)
