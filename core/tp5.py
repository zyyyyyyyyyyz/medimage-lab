"""TP5 FIRE retinal registration extracted from the authoritative notebook.

Source: ``TP_FIRE_Recalage_Complet_Kaggle_CORRIGE.ipynb`` cells 4, 6, 8,
10 and 12. Ground-truth landmarks are accepted only after feature-based
RANSAC estimation and are never passed to ``estimate_transform``.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import cv2
import numpy as np


MODELS = ("similarity", "affine", "homography")


class RegistrationError(ValueError):
    """Expected registration failure that can be shown safely in the UI."""


@dataclass(frozen=True)
class RegistrationResult:
    fixed: np.ndarray
    moving: np.ndarray
    registered: np.ndarray
    transform_processed: np.ndarray
    transform_full: np.ndarray
    detector: str
    n_matches: int
    n_inliers: int
    overlap: np.ndarray
    overlap_fraction: float
    before_metrics: dict[str, float]
    after_metrics: dict[str, float]
    matches_image: np.ndarray
    before_overlay: np.ndarray
    after_overlay: np.ndarray
    absolute_difference: np.ndarray
    landmark_metrics: dict[str, float] | None
    landmark_image: np.ndarray | None


def _validate_color_image(image: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or array.size == 0:
        raise RegistrationError(f"{label}: image couleur RGB non vide attendue")
    if array.shape[0] < 8 or array.shape[1] < 8:
        raise RegistrationError(f"{label}: dimensions insuffisantes")
    if not np.isfinite(array).all():
        raise RegistrationError(f"{label}: valeurs non finies")
    return np.clip(array, 0, 255).astype(np.uint8)


def resize_with_scale(image: np.ndarray, max_side: int = 850):
    """Notebook cell 4 resize and exact integer-derived scale matrix."""
    if int(max_side) < 64:
        raise ValueError("max_side doit être au moins 64")
    height, width = image.shape[:2]
    factor = min(1.0, float(max_side) / max(height, width))
    dst_w = max(1, int(round(width * factor)))
    dst_h = max(1, int(round(height * factor)))
    small = (
        cv2.resize(image, (dst_w, dst_h), interpolation=cv2.INTER_AREA)
        if factor < 1
        else image.copy()
    )
    scale = np.diag([dst_w / width, dst_h / height, 1.0]).astype(np.float64)
    return small, scale


def preprocess_for_features(rgb: np.ndarray) -> np.ndarray:
    """CLAHE on the green channel, exactly as TP5 cell 6."""
    green = rgb[:, :, 1]
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(green)


def match_features(fixed: np.ndarray, moving: np.ndarray, ratio: float = 0.78):
    """Return moving and fixed coordinates after the notebook Lowe test."""
    fixed_gray = preprocess_for_features(fixed)
    moving_gray = preprocess_for_features(moving)
    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.012)
        norm = cv2.NORM_L2
        detector_name = "SIFT"
    else:
        detector = cv2.ORB_create(nfeatures=6000, fastThreshold=8)
        norm = cv2.NORM_HAMMING
        detector_name = "ORB"
    keypoints_fixed, descriptors_fixed = detector.detectAndCompute(fixed_gray, None)
    keypoints_moving, descriptors_moving = detector.detectAndCompute(moving_gray, None)
    if (
        descriptors_fixed is None
        or descriptors_moving is None
        or len(descriptors_fixed) < 2
        or len(descriptors_moving) < 2
    ):
        empty = np.empty((0, 2), np.float32)
        return empty, empty.copy(), 0, detector_name
    matches = cv2.BFMatcher(norm).knnMatch(descriptors_moving, descriptors_fixed, k=2)
    good = [
        pair[0]
        for pair in matches
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance
    ]
    if not good:
        empty = np.empty((0, 2), np.float32)
        return empty, empty.copy(), 0, detector_name
    points_moving = np.float32([keypoints_moving[m.queryIdx].pt for m in good])
    points_fixed = np.float32([keypoints_fixed[m.trainIdx].pt for m in good])
    return points_moving, points_fixed, len(good), detector_name


def estimate_transform(src: np.ndarray, dst: np.ndarray, model: str, seed: int = 42):
    """Estimate a moving-to-fixed matrix with the exact TP5 RANSAC settings."""
    if model not in MODELS:
        raise ValueError(f"Modèle non reconnu: {model}")
    src = np.asarray(src, dtype=np.float32).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float32).reshape(-1, 2)
    if len(src) != len(dst):
        raise ValueError("Les listes de points source et destination doivent correspondre")
    minimum = 4 if model == "homography" else 3
    if len(src) < minimum:
        return None, 0
    cv2.setRNGSeed(seed)
    try:
        if model == "similarity":
            affine, inliers = cv2.estimateAffinePartial2D(
                src,
                dst,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.5,
                maxIters=5000,
                confidence=0.995,
                refineIters=15,
            )
        elif model == "affine":
            affine, inliers = cv2.estimateAffine2D(
                src,
                dst,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.5,
                maxIters=5000,
                confidence=0.995,
                refineIters=15,
            )
        else:
            homography, inliers = cv2.findHomography(
                src, dst, cv2.RANSAC, 4.0, maxIters=5000, confidence=0.995
            )
            if homography is None or not np.isfinite(homography).all():
                return None, 0
            return homography.astype(np.float64), int(inliers.sum()) if inliers is not None else 0
    except cv2.error:
        return None, 0
    if affine is None or not np.isfinite(affine).all():
        return None, 0
    matrix = np.vstack([affine, [0.0, 0.0, 1.0]])
    return matrix, int(inliers.sum()) if inliers is not None else 0


def warp_to_fixed(moving: np.ndarray, fixed: np.ndarray, transform: np.ndarray):
    width, height = fixed.shape[1], fixed.shape[0]
    return cv2.warpPerspective(
        moving,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def valid_retina_mask(rgb: np.ndarray) -> np.ndarray:
    """Notebook nonblack fundus ROI; this is not a clinical segmentation."""
    return (np.max(rgb, axis=2) > 18).astype(np.uint8)


def paired_overlap_mask(fixed: np.ndarray, moving: np.ndarray, transform: np.ndarray):
    roi_fixed = valid_retina_mask(fixed).astype(np.uint8)
    roi_moving = valid_retina_mask(moving).astype(np.uint8)
    warped_roi = cv2.warpPerspective(
        roi_moving,
        transform,
        (fixed.shape[1], fixed.shape[0]),
        flags=cv2.INTER_NEAREST,
    )
    overlap = (roi_fixed > 0) & (warped_roi > 0)
    coverage = float(overlap.sum() / max(int(roi_fixed.sum()), 1))
    return overlap, coverage


def similarity_metrics(
    fixed: np.ndarray, registered: np.ndarray, overlap: np.ndarray, bins: int = 32
):
    """MSE, NCC and NMI on identical green-channel overlap coordinates."""
    overlap = np.asarray(overlap, dtype=bool)
    if overlap.shape != fixed.shape[:2] or registered.shape != fixed.shape:
        raise ValueError("Images et masque de recouvrement non alignés")
    if overlap.sum() < 500:
        return {"mse": np.nan, "ncc": np.nan, "nmi": np.nan}
    fixed_values = fixed[:, :, 1][overlap].astype(np.float64)
    moving_values = registered[:, :, 1][overlap].astype(np.float64)
    mse = float(np.mean((fixed_values - moving_values) ** 2))
    std_fixed, std_moving = fixed_values.std(), moving_values.std()
    ncc = (
        float(
            np.mean(
                ((fixed_values - fixed_values.mean()) / std_fixed)
                * ((moving_values - moving_values.mean()) / std_moving)
            )
        )
        if std_fixed > 1e-12 and std_moving > 1e-12
        else np.nan
    )
    histogram, _, _ = np.histogram2d(
        fixed_values, moving_values, bins=bins, range=[[0, 256], [0, 256]]
    )
    probability = histogram / max(histogram.sum(), 1)
    px, py = probability.sum(axis=1), probability.sum(axis=0)

    def entropy(values):
        nonzero = values > 0
        return float(-np.sum(values[nonzero] * np.log(values[nonzero])))

    hx, hy, hxy = entropy(px), entropy(py), entropy(probability)
    mutual_information = max(0.0, hx + hy - hxy)
    nmi = float(2.0 * mutual_information / (hx + hy)) if hx + hy > 0 else np.nan
    return {"mse": mse, "ncc": ncc, "nmi": nmi}


def transform_points(points: np.ndarray, transform: np.ndarray):
    return cv2.perspectiveTransform(
        np.asarray(points, np.float64).reshape(-1, 1, 2),
        np.asarray(transform, np.float64),
    ).reshape(-1, 2)


def parse_landmarks(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Parse FIRE rows: x_fixed y_fixed x_moving y_moving."""
    if not payload:
        raise ValueError("Fichier de landmarks vide")
    try:
        values = np.loadtxt(BytesIO(payload), dtype=np.float64)
    except (ValueError, OSError) as exc:
        raise ValueError("Landmarks illisibles; quatre colonnes numériques attendues") from exc
    values = np.atleast_2d(values)
    if values.shape[1] != 4 or len(values) < 3 or not np.isfinite(values).all():
        raise ValueError(
            f"Landmarks mal formés: forme {values.shape}; au moins 3 lignes et 4 colonnes attendues"
        )
    return values[:, :2].copy(), values[:, 2:].copy()


def draw_feature_matches(fixed: np.ndarray, moving: np.ndarray, max_draw: int = 60):
    """Notebook cell 12 visualization, including its display detector limits."""
    moving_gray = preprocess_for_features(moving)
    fixed_gray = preprocess_for_features(fixed)
    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=3500, contrastThreshold=0.012)
        norm = cv2.NORM_L2
    else:
        detector = cv2.ORB_create(nfeatures=4500, fastThreshold=8)
        norm = cv2.NORM_HAMMING
    kp_moving, desc_moving = detector.detectAndCompute(moving_gray, None)
    kp_fixed, desc_fixed = detector.detectAndCompute(fixed_gray, None)
    if desc_moving is None or desc_fixed is None:
        raise RegistrationError("Descripteurs non détectés sur cette paire")
    raw = cv2.BFMatcher(norm).knnMatch(desc_moving, desc_fixed, k=2)
    matches = [
        first
        for pair in raw
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < 0.78 * second.distance
    ]
    matches.sort(key=lambda item: item.distance)
    canvas = cv2.drawMatches(
        moving_gray,
        kp_moving,
        fixed_gray,
        kp_fixed,
        matches[:max_draw],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return canvas, len(matches)


def residual_optical_flow(fixed: np.ndarray, registered: np.ndarray, optical_side: int = 440):
    """Exact TP5 Farnebäck residual-flow visualization and indicative remap."""
    scale = min(1.0, float(optical_side) / max(fixed.shape[:2]))
    shape = (int(fixed.shape[1] * scale), int(fixed.shape[0] * scale))
    fixed_small = cv2.resize(fixed, shape)
    registered_small = cv2.resize(registered, shape)
    fixed_gray = preprocess_for_features(fixed_small)
    registered_gray = preprocess_for_features(registered_small)
    flow = cv2.calcOpticalFlowFarneback(
        fixed_gray, registered_gray, None, 0.5, 3, 21, 4, 7, 1.5, 0
    )
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*magnitude.shape, 3), np.uint8)
    hsv[..., 0] = np.uint8(angle * 180 / np.pi / 2)
    hsv[..., 1] = 255
    scale_value = max(float(np.quantile(magnitude, 0.98)), 1e-4)
    hsv[..., 2] = np.uint8(np.clip(magnitude / scale_value * 255, 0, 255))
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    grid_y, grid_x = np.mgrid[0 : flow.shape[0], 0 : flow.shape[1]]
    refined = cv2.remap(
        registered_small,
        (grid_x + flow[..., 0]).astype(np.float32),
        (grid_y + flow[..., 1]).astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return flow_rgb, refined, float(np.mean(magnitude)), registered_small


def _landmark_evaluation(
    fixed: np.ndarray,
    scale_fixed: np.ndarray,
    transform_full: np.ndarray,
    reference_points: np.ndarray,
    moving_points: np.ndarray,
):
    predicted = transform_points(moving_points, transform_full)
    if not np.isfinite(predicted).all():
        raise RegistrationError("Projection des landmarks non finie")
    errors = np.linalg.norm(predicted - reference_points, axis=1)
    before = np.linalg.norm(moving_points - reference_points, axis=1)
    metrics = {
        "tre_before_px": float(np.mean(before)),
        "tre_after_px": float(np.mean(errors)),
        "tre_median_px": float(np.median(errors)),
        "n_landmarks": int(len(reference_points)),
    }
    reference_processed = transform_points(reference_points, scale_fixed)
    predicted_processed = transform_points(predicted, scale_fixed)
    canvas = fixed.copy()
    for expected, actual in zip(reference_processed, predicted_processed):
        expected_point = tuple(np.rint(expected).astype(int))
        actual_point = tuple(np.rint(actual).astype(int))
        cv2.circle(canvas, expected_point, 5, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.drawMarker(canvas, actual_point, (255, 0, 0), cv2.MARKER_TILTED_CROSS, 11, 2)
        cv2.line(canvas, actual_point, expected_point, (255, 255, 0), 1, cv2.LINE_AA)
    return metrics, canvas


def register_pair(
    fixed_full: np.ndarray,
    moving_full: np.ndarray,
    model: str = "affine",
    max_side: int = 850,
    landmarks: tuple[np.ndarray, np.ndarray] | None = None,
) -> RegistrationResult:
    """Run the TP5 single-pair pipeline; landmarks are evaluation-only."""
    fixed_full = _validate_color_image(fixed_full, "Image fixe")
    moving_full = _validate_color_image(moving_full, "Image mobile")
    fixed, scale_fixed = resize_with_scale(fixed_full, max_side)
    moving, scale_moving = resize_with_scale(moving_full, max_side)
    points_moving, points_fixed, n_matches, detector = match_features(fixed, moving)
    transform, n_inliers = estimate_transform(points_moving, points_fixed, model)
    minimum = 4 if model == "homography" else 3
    if transform is None or n_inliers < minimum:
        raise RegistrationError(
            f"Échec RANSAC ({model}): {n_matches} correspondance(s), {n_inliers} inlier(s); "
            f"minimum {minimum} requis"
        )
    transform_full = np.linalg.inv(scale_fixed) @ transform @ scale_moving
    if not np.isfinite(transform_full).all():
        raise RegistrationError("Transformation dans le repère original non finie")
    registered = warp_to_fixed(moving, fixed, transform)
    overlap, coverage = paired_overlap_mask(fixed, moving, transform)
    after_metrics = similarity_metrics(fixed, registered, overlap)

    moving_for_before = moving
    if moving.shape[:2] != fixed.shape[:2]:
        moving_for_before = cv2.resize(moving, (fixed.shape[1], fixed.shape[0]))
    common = (
        (valid_retina_mask(fixed) > 0)
        & (valid_retina_mask(moving_for_before) > 0)
        & (valid_retina_mask(registered) > 0)
    )
    before_metrics = similarity_metrics(fixed, moving_for_before, common)
    # Match the notebook's fair before/after comparison on the same coordinates.
    after_metrics = similarity_metrics(fixed, registered, common)

    matches_image, _ = draw_feature_matches(fixed, moving)
    before_overlay = cv2.addWeighted(fixed, 0.5, moving_for_before, 0.5, 0)
    after_overlay = cv2.addWeighted(fixed, 0.5, registered, 0.5, 0)
    difference = cv2.absdiff(fixed, registered)

    landmark_metrics = None
    landmark_image = None
    if landmarks is not None:
        reference_points, moving_points = landmarks
        if reference_points.shape != moving_points.shape:
            raise ValueError("Landmarks fixes et mobiles non appariés")
        landmark_metrics, landmark_image = _landmark_evaluation(
            fixed,
            scale_fixed,
            transform_full,
            reference_points,
            moving_points,
        )

    return RegistrationResult(
        fixed=fixed,
        moving=moving,
        registered=registered,
        transform_processed=transform,
        transform_full=transform_full,
        detector=detector,
        n_matches=n_matches,
        n_inliers=n_inliers,
        overlap=overlap,
        overlap_fraction=coverage,
        before_metrics=before_metrics,
        after_metrics=after_metrics,
        matches_image=matches_image,
        before_overlay=before_overlay,
        after_overlay=after_overlay,
        absolute_difference=difference,
        landmark_metrics=landmark_metrics,
        landmark_image=landmark_image,
    )
