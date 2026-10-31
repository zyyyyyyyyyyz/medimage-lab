# Reusable code review

## What is reusable now

| Source | Reusable surface | Status | Required action before backend use |
|---|---|---|---|
| `core/algorithms.py` | `ensure_gray`, `preprocess`, `clean_mask`, `segmentation`, `scores`, `overlay`, timing | Prototype-ready, not scientifically validated | Split pure algorithms from display helpers; type/shape validation; document each adaptation; add deterministic golden tests |
| `core/unet_tp4.py` | exact U-Net architecture; TP-like preprocessing; local load/predict wrappers | Architecture reusable; inference disabled without weights | Add model manifest/checksum, explicit capability status, preprocessing version, output-shape checks, and checkpoint tests |
| Executed TP3 row 6 | `classe_sombre`, `nettoyer`, `dice`, `iou`; RGB Mean Shift/K-means workflow | Good reference, notebook-coupled | Extract pure functions only after choosing whether parity or prototype behavior is authoritative |
| Executed FIRE row 5 | dataset discovery, feature matching, transforms, warping, TRE/similarity evaluation, summaries | Strongest modular notebook code, but outside current MVP | Extract later as an optional registration package with fixtures and OpenCV-version regression tests |
| `reference_code/*.py` | archival code-cell transcription | Reference only | Do not import: includes notebook magics, top-level I/O, plotting, hard-coded paths, and missing data |
| Empty template notebooks | none | Not reusable code | Preserve as teaching material only |

## Exact implemented pipelines

### Preprocessing and filtering

The unexecuted TP2 source implements Gaussian noise (`mean=0`, `std=25`), rank mean with `disk(3)`, median with `disk(3)`, Gaussian `sigma=1` with `preserve_range=True`, manual threshold 127, Otsu, 5×5 rectangular erosion/dilation/opening/closing, connected-component filtering, Sobel, Laplacian, morphological gradient, and OpenCV watershed. The unexecuted multi-modal atelier adds salt-and-pepper probabilities 0.005 each, 5×5 mean/median/Gaussian filters, PSNR, modality thresholds 110/105/150, and Otsu.

There is a likely bug in the atelier’s `filtre_gaussien`: `skimage.filters.gaussian(image, sigma=sigma)` is called without `preserve_range=True`, then clipped to 0–255 and cast to uint8. With ordinary uint8 input, scikit-image normally scales to 0–1, so most values collapse to 0 or 1. This code is unexecuted and must not be promoted unchanged.

The prototype implements a separate, simplified API: median/OpenCV, Gaussian/OpenCV, bilateral `(d=7, sigmaColor=45, sigmaSpace=45)`, CLAHE `(clipLimit=2, 8×8)`, and global equalization. Only median/Gaussian have direct inspected TP filtering antecedents; bilateral/global equalization are additions, and CLAHE’s inspected executed antecedent is FIRE feature preprocessing.

### Segmentation

Inspected executed TP3 exact behavior:

- resize RGB input with Pillow thumbnail to a 256 maximum side;
- Mean Shift: median filter `(3,3,1)`, RGB pixels, `estimate_bandwidth(quantile=.2, n_samples<=1000, random_state=0)`, minimum bandwidth 1, `bin_seeding=True`;
- K-means: RGB pixels, 3 clusters, `random_state=0`, `n_init=10`;
- choose the cluster with lowest mean `rgb2gray` intensity;
- retain only the largest connected component and fill holes;
- compute per-image Dice and IoU, returning 1 for two empty masks.

Prototype behavior differs: grayscale clustering, configurable 2–6 K-means clusters, seed 42, a 1,500-pixel Mean Shift bandwidth sample from a seeded generator, minimum bandwidth 5, maximum side 256/512 by method, dark/light choice, and optional component area filtering. Otsu/manual threshold, Chan–Vese, and watershed share the same binary-mask API, but they should be documented as adapted/added implementations, not all as exact executed TP3 code.

Prototype Chan–Vese uses normalized `[0,1]` grayscale, `mu=.25`, `lambda1=lambda2=1`, tolerance `.001`, 150 iterations, `dt=.5`, and checkerboard initialization. Prototype watershed uses 25th/75th intensity percentiles as markers and a Sobel elevation image. The unexecuted TP2 watershed instead constructs OpenCV markers from contour/morphological operations; they are not equivalent.

### Registration

The full executed FIRE notebook provides reusable functions with clear separation between estimation and evaluation:

- `find_fire_pairs` discovers image pairs and landmark text files; `read_pair` parses `x_ref,y_ref,x_test,y_test` and tracks resize matrices.
- `preprocess_for_features` applies CLAHE to the green channel.
- `match_features` prefers SIFT (`nfeatures=5000`, `contrastThreshold=.012`) and falls back to ORB (`nfeatures=6000`, `fastThreshold=8`), then applies a 0.78 ratio test from moving to fixed.
- `estimate_transform` supports similarity, affine, and homography using seeded RANSAC; `warp_to_fixed` uses perspective warping.
- Landmarks enter only after transform estimation. TRE is computed in original pixels after undoing processing resize.
- MSE/NCC/NMI are evaluated on identical valid overlapping green-channel pixels, with overlap coverage reported.
- `summarize_results` defines success as `status == ok` and TRE below 50 pixels, with all pairs in the denominator.

The missing manifest-listed FIRE example may overlap this implementation, but its exact source cannot be asserted.

### U-Net

`simple_unet_model` is the exact inspected TP4 architecture: 256×256×1 input in use; two 3×3 ReLU/he-normal convolutions per level; filters 16/32/64/128/256; dropout .1/.1/.2/.2/.3 down and .2/.2/.1/.1 up; four max-pools; transposed-convolution decoder with skip concatenation; and one-channel sigmoid output. Saved summary reports 1,940,817 trainable parameters.

Training code compiles with Adam, binary cross-entropy, and pixel accuracy; batch size 16; historical runs of 10 and 50 epochs. The 10-epoch call sets `shuffle=False`; the 50-epoch call uses the framework default. These runs do not establish reproducibility because package versions, checkpoint files, dataset snapshot, and independent test split are absent.

`preprocess_busi_like_tp` correctly reproduces the unusual axis-1 L2 normalization after resizing and before channel expansion. It first casts input to uint8, which may destroy higher-bit-depth inputs; the API must reject or explicitly window/convert such inputs rather than silently truncate.

## Dependencies

Prototype runtime declarations: Streamlit, NumPy, Pillow, headless OpenCV, SciPy, scikit-image, scikit-learn, and pandas; optional TensorFlow. Notebook-only dependencies also include matplotlib, imageio, IPython, Google Colab integration, and the standard library. Tests require pytest but it is not declared in a dev requirements file.

Current declarations are bounded but not locked and do not record Python/platform constraints or hashes. Recommended reproducibility artifacts:

1. `pyproject.toml` with runtime, `dl`, `test`, and `notebook` extras;
2. a lock/constraints file generated for supported Python versions;
3. a container image with exact system/OpenCV dependencies;
4. frontend `package-lock.json` committed after React/Vite work begins;
5. model and dataset manifests containing source, license, version/hash, modality, split, preprocessing identifier, and checkpoint checksum.

## Do not reuse as-is

- saved notebook scores as product claims;
- hard-coded Colab/Kaggle/Drive paths;
- top-level plotting and data discovery from archival `.py` transcriptions;
- automatic conversion of arbitrary medical pixels to uint8;
- filename-only checkpoint discovery;
- pixel accuracy as the main segmentation metric;
- cross-dataset or cross-modality score tables.
