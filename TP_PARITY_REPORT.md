# TP-to-application parity report

Validation date: 2026-10-31. The authoritative sources are the notebooks in `MedImage_Complete_11_Notebooks.zip`. Original notebook files were not modified. “Cell” refers to the notebook cell index recorded in the recovered code transcription, not an execution-order number.

## Classification rules

- **Exact**: same operation, parameters, input interpretation and output interpretation.
- **Numerically equivalent**: refactored implementation with equivalent array result for the tested domain.
- **Adapted**: traceable to a TP, but workflow, controls, display or postprocessing differ and are explicitly identified.
- **New**: no equivalent operation was found in the supplied TPs.

## TP1 — introduction and manipulation

| Application operation | Source | Classification | Parity/deviation |
|---|---|---|---|
| Raster loading and RGB/grayscale conversion | `01_Introduction_traitement_d'images_medicales_Mme_ATTIA.ipynb`, cells 10, 12 and 16 | Adapted | TP1 uses OpenCV/imageio paths. `core.io_utils.load_rgb` uses Pillow to validate uploaded PNG/JPEG/TIFF content and limits; `ensure_gray` uses OpenCV RGB→gray. The web validation is new safety behavior. |
| Dimensions and dtype display | TP1 cell 18 | Numerically equivalent | Same array shape/dtype concepts, presented as Streamlit metrics. |
| Histogram | TP1 cells 22 and 24 | Adapted | TP1 plots 256 bins; the web preview uses 64 bins for compact interactive display. It is not claimed as an exact TP plot. |
| Mask overlay | TP1 cells 34 and 36 | Adapted | TP1 replaces masked pixels with red. The web helper adds a translucent green fill and red boundary for comparison. This is a display extension, not an algorithmic TP result. |
| Video handling | TP1 cells 29 and 31 | Not integrated | The web application does not claim video support. |

## TP2 — preprocessing

| Application operation | Source | Classification | Preserved behavior |
|---|---|---|---|
| Median filter | `02-Pretraitement_images_medicales.ipynb`, cell 7 | Numerically equivalent | `skimage.filters.median` with `disk(3)`. The TP applies it to a noise-corrupted exercise image; the web applies the same operator to the uploaded image. |
| Gaussian filter | TP2 cell 7 | Numerically equivalent | `skimage.filters.gaussian`, `sigma=1.0`, `preserve_range=True`, clip and cast to uint8. |
| Manual threshold | TP2 cell 10 | Exact at default; adapted control | `cv2.threshold(..., 50, 255, THRESH_BINARY)`. Web default is restored to 50; changing the slider is an explicit user adaptation. |
| Otsu | TP2 cell 12 | Exact | `cv2.THRESH_BINARY + cv2.THRESH_OTSU`; bright-class output interpretation is preserved. |
| Erosion/dilation/opening/closing | TP2 cells 18 and 20 | Not exposed | No claim of web parity. |
| Sobel/Laplacian/morphological gradient | TP2 cells 25 and 27 | Not exposed | No claim of web parity. |
| Watershed | TP2 cells 32 and 34 | **Adapted web extension** | TP2 builds OpenCV markers from contours/ground truth. The web uses automatic 25th/75th percentile markers with scikit-image. It is visibly labelled non-equivalent and must not be cited as TP2 watershed. |

Regression evidence: `test_tp2_filter_numerical_parity` compares complete arrays to direct scikit-image calls; `test_tp2_threshold_numerical_parity` compares complete arrays and the Otsu threshold to direct OpenCV calls.

## TP3 — BAGLS conventional segmentation

The web routes TP3 methods only through `core.tp3`; the earlier grayscale substitutes were removed from the generic algorithm module.

| Operation | Source notebook/cell | Classification | Preserved behavior |
|---|---|---|---|
| Working image | `03-Segmentation_Mme_ATTIA.ipynb`, cells 9/14 | Exact | RGB Pillow image, `thumbnail((256,256))`, BAGLS aspect ratio retained. |
| Mean Shift | TP3 cell 9; duplicate executed variant cell 6 | Exact | RGB median filter `(3,3,1)`; RGB pixels; bandwidth quantile 0.2; at most 1000 samples; `random_state=0`; minimum bandwidth 1; `bin_seeding=True`. |
| K-means | TP3 cell 14; duplicate executed variant cell 11 | Exact | RGB pixels; exactly 3 clusters; `random_state=0`; `n_init=10`. |
| Dark-class selection | TP3 cell 18; duplicate variant cell 14, `classe_sombre` | Exact | Mean `rgb2gray` intensity for each label; minimum mean selected. The web does not expose a light-class switch for TP3. |
| Binary cleanup | TP3 cell 18, `nettoyer` | Exact | Connected components, retain the largest non-background component, then `binary_fill_holes`. Generic web area cleanup is not applied to TP3 results. |
| Dice and IoU | TP3 cell 20; duplicate variant cell 15, `dice`/`iou` | Exact | Boolean masks; same empty-pair policy and shape rejection. |
| Reference mask | TP3 cell 20 | Exact | Original image/mask dimensions must match; reference is resized to the 256-side working image with nearest-neighbor interpolation. |
| Chan–Vese | TP3 cell 24 | Exact | `rgb2gray`; `mu=.25`, `lambda1=lambda2=1`, `tol=.001`, 200 iterations, `dt=.5`, checkerboard; dark region plus TP3 cleanup. |

Regression evidence on the identical bundled BAGLS image/mask:

| Method | Notebook saved output | Refactored module |
|---|---:|---:|
| Mean Shift Dice / IoU | 0.8337 / 0.7148 | 0.8337 / 0.7148 |
| K-means Dice / IoU | 0.8301 / 0.7095 | 0.8301 / 0.7095 |

`test_tp3_recorded_bagls_regression` enforces those values to four decimals. `test_tp3_chan_vese_is_deterministic_and_binary` verifies deterministic full-mask equality across repeated calls, shape and binary interpretation. No saved Chan–Vese numeric result was used as fabricated evidence.

## TP4 — BUSI U-Net

| Operation | Source | Classification | Preserved behavior |
|---|---|---|---|
| `simple_unet_model(256,256,1)` | `04_UNET_Mme_ATTIA_(1) (3).ipynb`, cell 7 | Exact | Same convolution, dropout, pooling, transpose-convolution, skip connections and sigmoid output. |
| BUSI image/mask loading | TP4 cell 18 | Exact in reproduction notebook | Grayscale image; default OpenCV 256×256 resize; all `_mask*.png` masks resized nearest-neighbor and merged with `np.maximum`; normal cases excluded by default. |
| Image normalization | TP4 cell 20 | Exact with TensorFlow; numerically equivalent fallback | `tensorflow.keras.utils.normalize(image_dataset, axis=1)` before channel expansion. Inference uses that exact call when TensorFlow exists; the TensorFlow-free test fallback computes the same L2 rule. |
| Mask binarization | TP4 cell 23 | Exact in reproduction notebook | Channel expansion then `>127`, float32. |
| Split | TP4 cell 27 | Exact in reproduction notebook | 90/10, `random_state=42`; no invented split. |
| 10-epoch training | TP4 cell 29 | Exact in reproduction notebook | Adam, binary cross-entropy, accuracy, batch 16, 10 epochs, `shuffle=False`, original checkpoint name. |
| 50-epoch training | TP4 cell 38 | Exact in reproduction notebook | New model, batch 16, 50 epochs, default Keras shuffle, original checkpoint name. |
| Prediction threshold | TP4 cells 33 and 38 | Exact | Fixed strict `prediction > 0.5`; the web does not expose an alternative threshold. |
| Evaluation | TP4 cells 31, 33, 36 and 38 | Exact in reproduction notebook | Validation loss/accuracy, pooled IoU and mean per-image Dice; no cross-dataset comparison. |
| Return mask at upload dimensions | No notebook equivalent | Adapted postprocessing | Nearest-neighbor resize is only for web display/export after 256×256 prediction. |
| Checkpoint loading/shape checks | No notebook equivalent | New safety integration | `compile=False`; input and output must be `(256,256,1)`. |

`test_tp4_normalization_equivalence` verifies the exact default-linear resize, axis-1 L2 normalization and channel expansion numerically. `test_tp4_prediction_keeps_probability_and_uses_strict_threshold` verifies that `0.5` remains background and only values strictly above it become foreground. The supplied checkpoint was loaded with TensorFlow 2.20.0 and produced a finite real prediction with validated input/output shapes. This is an integration smoke test, not reproduction of the notebook's historical metrics.

`training/TP4_UNET_GPU_REPRODUCTION.ipynb` is an unexecuted reproduction notebook, not an altered original. It contains the original data construction, configurations, evaluation and saving names:

- `unet_breast_batch16_epochs10.hdf5`
- `unet_breast_batch16_epochs50.hdf5`

## FIRE — optional registration

The executed FIRE notebook’s cell 6 contains green-channel CLAHE, SIFT/ORB matching, RANSAC similarity/affine/homography estimation and warping; cells 8/10 contain overlap, similarity and landmark/TRE evaluation. Registration is not exposed by the current web workflow and no parity claim is made.

The general CLAHE preview uses the same `clipLimit=2.0`, `tileGridSize=(8,8)` operation as FIRE cell 6, but applying it as a general preprocessor is an **adapted reuse**, not FIRE registration.

## Explicit additions without a direct TP equivalent

- Bilateral filter (`d=7`, sigma color/space 45).
- Global histogram equalization as an interactive preprocessing option.
- Automatic percentile-marker watershed.
- Upload byte/pixel/content validation.
- Green translucent overlay plus red boundary.
- PNG/CSV downloads, progress UI, traceability panel and checkpoint upload/configuration.

These additions are labelled in the interface and are not presented as original TP algorithms. No nnU-Net, SAM, MedSAM, MONAI model, NILM dataset or other external segmentation model/dataset is included.

## Missing assets and remaining deviations

- The supplied `models/unet_breast_batch16_epochs50.hdf5` is present and compatible with the exact TP4 architecture; its SHA-256 is `0a2eed5f5893560710fe3fb0211cae554c66bf1a22390139a8d72db5cfb403bf`.
- TensorFlow 2.20.0 is installed only in the project `.venv`; the reproduction notebook remains unexecuted and no training is claimed.
- BUSI is not bundled; no training or BUSI metric was reproduced.
- The web offers a configurable TP2 manual threshold as an explicit adaptation; the TP4 U-Net threshold is fixed at the original strict `> 0.5`.
- TP1 histogram/overlay and web watershed are intentionally not pixel-identical to notebook visualizations.
- FIRE registration remains optional and not integrated.

## Test result

The parity suite plus baseline suite now collects 12 tests. The final verified run completed with all 12 passing in 8.08 seconds; see `FINAL_STATUS.md`. Streamlit startup and initial UI rendering completed with zero application exceptions, errors or warnings.
