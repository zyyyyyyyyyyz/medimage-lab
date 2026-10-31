# Final academic demonstration checklist

## Start once

From the project directory:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Keep this single browser session open. The sidebar provides **Laboratoire** (TP1–TP4), **TP5 · Recalage FIRE**, and **Guide scientifique**. This is an educational prototype, not a diagnostic device.

## TP1 — Introduction and image manipulation

1. Open **Laboratoire** and leave **Utiliser l'exemple BAGLS** enabled, or disable it and upload a PNG/JPEG/TIFF.
2. In **Aperçu & histogramme**, show the original image, dimensions, grayscale processing preview, aligned reference when present, and intensity histogram.
3. Explain that TP1 is the import/manipulation/visualization foundation; it has no standalone accuracy metric or dedicated export. Segmentation exports are demonstrated in TP2–TP4.

Recommended input: bundled `assets/bagls_test_2.png` through the BAGLS toggle.

## TP2 — Preprocessing and threshold segmentation

1. Under **Prétraitement**, demonstrate **Médian** (disk radius 3) and **Gaussien** (sigma 1). Bilateral and global histogram equalization are labelled web additions; CLAHE is adapted from FIRE.
2. Select **Seuil manuel** and **Otsu** under segmentation. The original manual threshold is 50.
3. Show processed previews, masks, overlays, execution times, and Dice/IoU with the bundled aligned BAGLS mask.
4. Download a binary mask PNG and the shared results CSV.

Recommended input: bundled BAGLS example and mask.

## TP3 — Conventional BAGLS segmentation

1. Select **K-means**, **Mean Shift**, and **Chan–Vese**.
2. Explain that these exact TP3 methods operate on the original RGB image resized with `thumbnail(256)`, independently of the exploratory preprocessing control.
3. Show feature-independent masks, overlays, execution times, Dice/IoU, PNG downloads, and the comparison CSV.
4. Mean Shift and Chan–Vese can take longer than thresholding; wait for the progress indicator.

Recommended input: bundled BAGLS pair. Do not combine these scores with BUSI results.

## TP4 — BUSI U-Net

1. Disable the BAGLS example, select **Échographie mammaire / BUSI**, and upload a genuine BUSI grayscale ultrasound.
2. Select **U-Net BUSI**. Confirm detection of `models/unet_breast_batch16_epochs50.hdf5`.
3. Show the original image, probability map, binary mask using strict `>0.5`, overlay, execution time, and mask PNG/CSV exports.
4. Upload a corresponding, pixel-aligned BUSI mask only if available; Dice/IoU must remain absent otherwise.

Required scientific input: a genuine BUSI image and its corresponding mask for metrics. No standalone BUSI image was found locally during final verification, so the final check used BAGLS only as a cross-modality tensor/export smoke test. That smoke test is not BUSI validation and no notebook historical metric was reproduced.

## TP5 — FIRE retinal registration

1. Choose **TP5 · Recalage FIRE** in the same sidebar session.
2. Upload the FIRE `<pair>_1` image as **fixed** and `<pair>_2` as **moving**.
3. Optionally upload `control_points_<pair>_1_2.txt`, formatted as `x_fixed y_fixed x_moving y_moving`. These points evaluate TRE only and never estimate the transformation.
4. Demonstrate Similarity, Affine, and Homography separately. Show feature matches, fixed/mobile/registered images, before/after overlays, absolute difference, MSE/NCC/NMI, and residual Farnebäck flow.
5. With landmarks, show TRE before/after and the landmark error visualization.
6. Download the registered PNG, result CSV, and moving-to-fixed transformation JSON.

Required scientific input: a genuine FIRE pair and matching annotation file. None was found locally during final verification. A geometrically derived pair from a retinal photograph exercised the complete pipeline and exports, but it is synthetic-pair engineering evidence, not genuine FIRE performance.

## Final verified state

- One-session navigation: Laboratoire → Guide scientifique → TP5 → Laboratoire, with zero application exceptions or errors.
- All six conventional methods: 15 visual elements, one comparison table, six mask downloads and one CSV download.
- TP4 checkpoint: loads successfully, produces finite probabilities and a valid binary PNG.
- TP5 derived-pair smoke: matching, RANSAC, metrics, TRE, residual flow, PNG/CSV/JSON exports all completed.
- Streamlit headless startup: HTTP 200.
- Dependency check: no broken requirements.
- Final regression suite: **20 passed in 90.36 seconds**.

Academic demonstration readiness: **ready**, with genuine BUSI and FIRE dataset validation explicitly pending presenter-supplied local inputs.
