import cv2
import numpy as np
import pytest

import core.tp5 as tp5


def _grid_points():
    x, y = np.meshgrid(np.linspace(30, 220, 5), np.linspace(25, 210, 5))
    return np.column_stack([x.ravel(), y.ravel()]).astype(np.float32)


@pytest.mark.parametrize(
    ("model", "matrix"),
    [
        (
            "similarity",
            np.array([[1.05, -0.18, 12.0], [0.18, 1.05, -7.0], [0.0, 0.0, 1.0]]),
        ),
        (
            "affine",
            np.array([[1.04, 0.12, 8.0], [-0.07, 0.96, 14.0], [0.0, 0.0, 1.0]]),
        ),
        (
            "homography",
            np.array([[1.02, 0.04, 9.0], [-0.03, 0.98, 11.0], [0.0002, -0.00015, 1.0]]),
        ),
    ],
)
def test_each_tp5_transform_type(model, matrix):
    source = _grid_points()
    destination = tp5.transform_points(source, matrix).astype(np.float32)
    estimated, inliers = tp5.estimate_transform(source, destination, model)
    assert estimated is not None
    assert inliers == len(source)
    predicted = tp5.transform_points(source, estimated)
    np.testing.assert_allclose(predicted, destination, atol=0.05)


def test_tp5_failure_handling_for_insufficient_matches(monkeypatch):
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    with pytest.raises(tp5.RegistrationError, match="Échec RANSAC"):
        tp5.register_pair(image, image, model="affine")
    matrix, inliers = tp5.estimate_transform(
        np.array([[0, 0], [1, 1]], np.float32),
        np.array([[2, 2], [3, 3]], np.float32),
        "affine",
    )
    assert matrix is None and inliers == 0


def test_tp5_similarity_metrics_and_small_overlap():
    y, x = np.mgrid[:64, :64]
    green = ((x * 5 + y * 3) % 256).astype(np.uint8)
    image = np.stack([green // 2, green, green // 3], axis=-1)
    overlap = np.ones((64, 64), dtype=bool)
    metrics = tp5.similarity_metrics(image, image.copy(), overlap)
    assert metrics["mse"] == 0.0
    assert metrics["ncc"] == pytest.approx(1.0)
    assert metrics["nmi"] == pytest.approx(1.0)
    invalid = tp5.similarity_metrics(image, image, np.zeros((64, 64), dtype=bool))
    assert all(np.isnan(value) for value in invalid.values())


def test_tp5_landmarks_are_evaluation_only(monkeypatch):
    rng = np.random.default_rng(7)
    image = rng.integers(20, 230, size=(160, 180, 3), dtype=np.uint8)
    points = np.array(
        [[20, 20], [150, 20], [20, 130], [150, 130], [90, 75]], dtype=np.float32
    )
    monkeypatch.setattr(
        tp5,
        "match_features",
        lambda fixed, moving: (points.copy(), points.copy(), len(points), "SIFT"),
    )
    monkeypatch.setattr(
        tp5,
        "draw_feature_matches",
        lambda fixed, moving: (np.zeros((50, 100, 3), np.uint8), len(points)),
    )
    reference = points.astype(np.float64)
    moving = points.astype(np.float64)
    without_gt = tp5.register_pair(image, image, model="affine")
    with_gt = tp5.register_pair(
        image, image, model="affine", landmarks=(reference, moving)
    )
    np.testing.assert_allclose(with_gt.transform_full, without_gt.transform_full, atol=1e-10)
    assert with_gt.landmark_metrics["tre_after_px"] == pytest.approx(0.0, abs=1e-8)
    assert with_gt.landmark_metrics["n_landmarks"] == 5
    assert with_gt.landmark_image.shape == image.shape


def test_tp5_residual_optical_flow_shapes():
    rng = np.random.default_rng(2)
    image = rng.integers(0, 255, size=(120, 160, 3), dtype=np.uint8)
    flow_rgb, refined, mean_magnitude, registered_small = tp5.residual_optical_flow(
        image, image.copy(), optical_side=100
    )
    assert flow_rgb.shape == refined.shape == registered_small.shape
    assert flow_rgb.shape[:2] == (75, 100)
    assert np.isfinite(mean_magnitude)
    assert mean_magnitude < 0.01


def test_tp5_landmark_parser():
    payload = b"10 20 11 21\n30 40 31 41\n50 60 51 61\n"
    fixed, moving = tp5.parse_landmarks(payload)
    assert fixed.shape == moving.shape == (3, 2)
    with pytest.raises(ValueError, match="4 colonnes"):
        tp5.parse_landmarks(b"1 2 3\n4 5 6\n7 8 9\n")
