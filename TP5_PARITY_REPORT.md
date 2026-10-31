# TP5 FIRE notebook-to-application checklist

Authoritative notebook inspected in full: `reference_notebooks/TP_FIRE_Recalage_Complet_Kaggle_CORRIGE (1).ipynb` (31 cells, no executed outputs). The requested unsuffixed `notebooks/` path was not present; the existing reference copy was read without modification.

## Web integration

| Notebook feature | Cell | Web status | Fidelity / scope |
|---|---:|---|---|
| Independent fixed/mobile resize and scale matrices | 4 | Implemented | Exact `min(1, max_side/max(h,w))`, rounded integer dimensions, `INTER_AREA`, and actual x/y ratios; web uses notebook `DEMO_SIDE=850`. |
| Green channel + CLAHE | 6 | Implemented | Exact `clipLimit=2.0`, `tileGridSize=(8,8)`. |
| SIFT / ORB fallback | 6 | Implemented | Exact SIFT `nfeatures=5000`, `contrastThreshold=0.012`; ORB `nfeatures=6000`, `fastThreshold=8`. |
| Lowe matching | 6 | Implemented | Moving-to-fixed BFMatcher, `k=2`, strict ratio `<0.78`. |
| Similarity RANSAC | 6 | Implemented | `estimateAffinePartial2D`, threshold 3.5, 5000 iterations, confidence 0.995, 15 refinement iterations, seed 42. |
| Affine RANSAC | 6 | Implemented | `estimateAffine2D` with the same parameters. |
| Homography RANSAC | 6 | Implemented | `findHomography`, threshold 4.0, 5000 iterations, confidence 0.995, seed 42. |
| Moving-to-fixed warp | 6 | Implemented | `warpPerspective`, linear interpolation, constant black border. |
| Retina overlap | 8 | Implemented | Exact nonblack threshold `max(channel)>18`; nearest-neighbor mask warp and coverage relative to fixed ROI. |
| MSE, NCC, NMI | 8, 10 | Implemented | Green channel, 32-bin joint histogram, at least 500 common valid pixels; before/after use identical coordinates. |
| Feature-match figure | 12 | Implemented | Exact display detector limits (SIFT 3500 / ORB 4500), ratio 0.78, distance sort, maximum 60 matches. |
| Original/fixed/mobile/registered views | 12 | Implemented | Includes before and after overlays and absolute RGB difference. |
| Optional FIRE landmarks and TRE | 4, 9, 10, 12 | Implemented | Four-column FIRE format; at least three rows. Landmarks are passed only to evaluation after RANSAC. TRE is in original uploaded-image pixels. |
| Residual Farnebäck flow | 12, 24–25 | Implemented | Exact 440-pixel visualization scale and parameters `(0.5,3,21,4,7,1.5,0)`, 98th-percentile HSV magnitude, and indicative remap. |
| Registered image / metrics / matrix downloads | Web addition | Implemented | PNG, one-row CSV and JSON moving-to-fixed matrices. |

## Notebook-only benchmark functionality

The following is intentionally not exposed in the single-pair web workflow:

- Kaggle input discovery and FIRE directory scanning (cells 4, 16–17).
- `FirePair` dataset orchestration and automatic category demo selection.
- Synthetic FIRE file generation (cells 13–15).
- Full 134-pair × 3-model benchmark, S/P/A aggregation, 50-pixel pedagogical threshold, plots and benchmark CSVs (cells 26–29).

These are dataset/benchmark utilities, not required for interactive two-upload registration. The web app does not claim to reproduce the official FIRE AUC or the notebook benchmark.

## Validation evidence

- Focused TP5 suite: 8 passed. It covers similarity, affine, homography, insufficient matches, metric validity, landmark parsing, landmark isolation from estimation, and residual-flow output.
- Complete TP1–TP5 suite: 20 passed in 9.05 seconds.
- Streamlit test harness: default page 0 exceptions/errors/warnings; TP5 page 0 exceptions, 0 errors, and one expected prompt warning before uploads.
- Derived retinal-image smoke test (not a genuine FIRE pair): all three transformations completed with 2,558 feature matches and approximately 2,520 inliers. This is engineering evidence only.

## Missing validation asset

No genuine local FIRE image pair with `control_points_*_1_2.txt` was found in the project or Downloads directory. Therefore no real FIRE-pair performance or TRE result is claimed. A genuine smoke test remains pending until a local FIRE pair is supplied; no dataset was downloaded.
