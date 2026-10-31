# MedImage Lab audit

Audit date: 2026-10-31 (Africa/Tunis). Scope: supplied prototype archive, the extracted prototype, its manifests, notebooks identifiable from those manifests, extracted Python, assets, model directory, and tests. This is an evidence audit only; no implementation or training was performed.

## Executive finding

The current package is a useful Streamlit teaching prototype, not yet a reproducible or clinically valid image-processing platform. It contains working-looking algorithm and U-Net architecture code, but **no trained U-Net checkpoint**. A model definition is not pretrained weights. Conventional methods are partly adapted from notebooks and partly new prototype code; they must not all be described as exact notebook extractions.

The prototype ZIP is internally consistent: all 22 archived files byte-match the extracted workspace. The separate input `Medical_TP_Notebooks_Complete_Updated.zip` was not found under `C:\Users\aziza\Downloads`, so its provenance and contents could not be independently verified. The prototype manifests describe 11 unique notebooks. Nine of those exact SHA-256 artifacts are currently readable (five bundled templates plus four loose executed notebooks); two manifest-listed executed artifacts are absent. See `NOTEBOOK_INVENTORY.md`.

## Evidence and boundaries

- `MedImage_Lab_Web_Prototype_Updated.zip`: present, 172,427 bytes; 22/22 files match the extracted workspace.
- `Medical_TP_Notebooks_Complete_Updated.zip`: absent from the searched Downloads tree.
- Original notebooks were read as JSON only and were not edited or executed.
- Existing saved cell outputs are historical evidence embedded in notebooks, not results reproduced during this audit.
- No datasets were downloaded, no model was trained, and no weights were downloaded.
- Only `assets/bagls_test_2.png` and `assets/bagls_test_2_mask.png` are bundled data examples.

## Phase 1B baseline status

The project root and its immediate parent were searched again on 2026-10-31. `Medical_TP_Notebooks_Complete_Updated.zip` is still absent, so `reference_materials/` was not created from an archive and no partial collection was presented as an extraction. Direct JSON inspection remains possible for nine exact-hash notebook artifacts. The other two are accounted for as missing manifest identities, not verified originals. This is the only unresolved Phase 1B acceptance blocker.

The environment blocker is resolved in a project-local Python 3.11 virtual environment. Exact packages are frozen in `requirements-baseline-lock.txt`; `pip check`, imports, all four tests, Streamlit script execution, and an HTTP startup smoke test succeeded. Details are in `ENVIRONMENT_SETUP.md` and `BASELINE_TEST_REPORT.md`.

## Prototype review

The Streamlit app supports PNG/JPEG/TIFF input; grayscale conversion; median, Gaussian, bilateral, CLAHE, and histogram equalization; Otsu, manual threshold, K-means, Mean Shift, Chan–Vese, and watershed; overlays; PNG/CSV export; and Dice/IoU when a reference mask exists. It limits decoded images to 12 million pixels and reduces costly methods. U-Net inference is gated by a user-selected BUSI modality and by the expected local file name.

Positive controls include no random-weight fallback, a BUSI-only warning, use of the original image rather than exploratory filtering for U-Net, nearest-neighbor mask resizing, binary outputs, and metrics only when a reference is supplied.

Important limitations and discrepancies:

- `core/unet_tp4.py:simple_unet_model` is an exact architecture transcription of the inspected TP4 cell. `preprocess_busi_like_tp`, checkpoint loading, and prediction are production adaptations, not literal notebook functions.
- The inspected executed TP3 uses RGB K-means/Mean Shift, `random_state=0`, and retains only the largest connected component before filling holes. The prototype clusters grayscale, uses seed 42, bounds/resamples computation, and optionally retains all components above an area threshold. Those changes may be sensible, but are not exact extraction.
- Otsu/manual threshold and much of morphology come from unexecuted TP2/atelier templates. Bilateral filtering and global histogram equalization are prototype additions in the inspected evidence. CLAHE is used by the executed FIRE registration notebook, not by the inspected TP3 segmentation.
- Exact provenance of prototype Chan–Vese cannot be fully verified because one manifest-listed executed TP3 artifact is missing. Prototype watershed is a new intensity-marker implementation and differs from the unexecuted TP2 OpenCV watershed exercise.
- The app accepts only display rasters and coerces them to 8-bit RGB/grayscale. It does not preserve DICOM/NIfTI metadata, bit depth, spacing, orientation, photometric interpretation, window/level, or rescale slope/intercept.
- A mismatched ground-truth mask is resized with a warning. This is convenient for a demo but cannot establish anatomical alignment; metrics may be misleading.
- Modality is self-declared by the user. The UI guard prevents an obvious wrong U-Net selection but does not verify acquisition protocol or BUSI compatibility.
- No upload byte limit, authentication, persistence policy, audit trail, DICOM de-identification, model checksum/provenance record, or resource isolation is implemented.
- The app and tests contain mojibake when viewed in the current shell, indicating an encoding/display portability issue even though source parsing succeeds.

## Test execution

The original global environment failure remains documented evidence: NumPy 2.4.6 was ABI-incompatible with its old scikit-learn 1.2.2 and pandas 1.5.3, so the first audit run collected zero tests. That global environment was not modified.

Phase 1B created `.venv` with Python 3.11.9 and an exact lock. In this isolated environment:

```text
pip check: No broken requirements found.
pytest: collected 4 items
test_scores_exact PASSED
test_methods_shapes PASSED
test_preprocessing_and_overlay PASSED
test_tp4_normalization_equivalence PASSED
4 passed, 1 warning in 3.22s (final verification run)
```

The warning is an unsuppressed scikit-learn convergence warning caused by asking for three clusters on a synthetic image with two distinct values. It does not represent a hidden failure of the asserted binary-mask/shape contract.

Streamlit’s application test harness completed with zero app exceptions, warnings, or errors and rendered the title and results dataframe. A real headless server on port 8765 returned HTTP 200 before deliberate Ctrl+C shutdown. In the later TP4 integration, TensorFlow 2.20.0 and the supplied compatible checkpoint were tested successfully; see `FINAL_STATUS.md`. No historical notebook metric was reproduced.

## Checkpoint verification

No `*.h5`, `*.hdf5`, `*.keras`, `*.pt`, `*.pth`, `*.onnx`, `*.ckpt`, or `*.safetensors` exists in the prototype tree. `models/` contains only `README.md`. A root-level Downloads file named `redd.h5` exists, but it is unrelated NILM data and has neither the expected name nor established model provenance; it is **not** a U-Net checkpoint.

The TP4 notebook has saved output showing that files named `unet_breast_batch16_epochs10.hdf5` and `unet_breast_batch16_epochs50.hdf5` were written to Google Drive during a historical run. The files themselves are not embedded in the notebook and are not supplied. Inference is therefore unavailable and all saved TP4 metrics remain notebook-recorded, unreproduced evidence.

## Scientific risk register

| Risk | Evidence | Required control |
|---|---|---|
| Cross-modality model use | U-Net trained on BUSI breast ultrasound; BAGLS is laryngeal endoscopy and FIRE is retinal fundus | Explicit model card and server-side modality allow-list; reject rather than merely warn for incompatible inference |
| Invalid comparisons | TP3 BAGLS, TP4 BUSI, and FIRE registration use different tasks/datasets | Compare methods only on the same task, modality, split, image set, preprocessing, and ground truth |
| Ground-truth misuse | App can resize a mask of different dimensions | Require aligned dimensions or an explicit, documented transform; never infer registration from resizing |
| Metric mismatch | TP4 IoU is pooled over the whole validation tensor while Dice is per-image mean; accuracy is background dominated | Report per-image Dice/IoU with distribution and empty-mask policy; optionally macro/micro separately; do not lead with pixel accuracy |
| Split leakage/bias | TP4 uses a single random 90/10 image split and no independent test set | Use patient/group-aware fixed train/validation/test splits where identifiers permit; document dataset version and seed |
| Normalization drift | TP4 uses `tensorflow.keras.utils.normalize(image_dataset, axis=1)`, not `/255` | Freeze the exact axis-1 L2 rule as a tested preprocessing contract; store it with model metadata |
| Mask construction drift | Multiple BUSI masks are merged with pixelwise maximum; normal cases are excluded | Preserve and document merge policy and class inclusion; evaluate the population the model will serve |
| Interpolation/bit-depth loss | App converts to uint8 and resizes | Preserve original pixel representation and metadata in the scientific layer; make display conversion separate |
| Notebook-output overclaim | Saved outputs exist but were not reproduced here | Label all such numbers “saved notebook output”; reproduce only after datasets, environment, and checkpoint are available |
| Registration metric misuse | FIRE correctly estimates without landmarks and evaluates afterward, but similarity metrics depend on overlap/green channel | Keep landmarks evaluation-only; report TRE in original pixels and overlap coverage; do not compare registration metrics to segmentation metrics |

## Decision

Phase 1B stops here. The conventional-processing environment and application baseline are ready. Phase 2 should not start until the missing notebook archive is supplied and the two manifest-only notebook identities are directly verified, or the project owner explicitly accepts them as unavailable evidence. U-Net must remain a disabled optional capability until authentic weights and provenance are supplied.
