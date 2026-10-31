# Notebook inventory

“Executed” means at least one saved output and/or non-null execution count exists in the notebook JSON. It does not mean this audit reran the notebook. Hashes are SHA-256 of the complete notebook file.

The 11-unique count below is taken from `docs/MANIFEST_COMPLET.json`. Because the named notebook ZIP is absent, rows marked “manifest only” could not be independently re-read.

## Phase 1B recovery result

`Medical_TP_Notebooks_Complete_Updated.zip` was not present in the project root, anywhere below it, or in the immediate parent directory on 2026-10-31. Extraction to `reference_materials/` was therefore impossible. No original notebook was copied, renamed, rewritten, or executed.

For every physically available candidate, the file was parsed directly as JSON and its code-cell `execution_count` and `outputs` arrays were counted. The five bundled source notebooks and four loose executed notebooks exactly match nine hashes in the existing inventory. Rows 2 and 3 remain manifest-only; their code/output counts below are explicitly unverified. Thus all 11 identities are accounted for, but only nine are recovered and directly classified. The recovery acceptance condition is not fully met.

| # | Manifest identity / hash prefix | Availability | Cells (code) | Saved execution evidence | Classification | Main scope |
|---:|---|---|---:|---|---|---|
| 1 | `01_Introduction_traitement_d'images_medicales_Mme_ATTIA.ipynb` / `a8c4f73f` | Loose file, exact hash | 37 (15) | 13 code cells have counts; 12 have outputs | Executed | raster/video reading, dimensions, histograms, channels, BAGLS overlay |
| 2 | `03-Segmentation_Mme_ATTIA.ipynb` / `57872823` | **Missing; manifest only** | 26 (8) | Manifest: 7 counted; 6 with outputs | Executed, unverified source | segmentation; reported functions `classe_sombre`, `nettoyer`, `dice`, `iou` |
| 3 | `TP_Recalage_Retine_FIRE_Exemple_Execute(1).ipynb` / `792f2bbb` | **Missing; manifest only** | 27 (14) | Manifest: 14 counted; 8 with outputs | Executed, unverified source | FIRE registration example |
| 4 | `04_UNET_Mme_ATTIA_(1) (3).ipynb` / `74a86bf1` | Loose file, exact hash | 40 (20) | 20 counted; 15 with outputs | Executed | BUSI loading, U-Net, training/evaluation code |
| 5 | `notebookef64fd0abd.ipynb` / `0986f4b8` | Loose file, exact hash | 31 (15) | 15 counted; 9 with outputs | Executed | full FIRE registration benchmark |
| 6 | `03_Segmentation_Mme_ATTIA(1).ipynb` / `361e7dd5` | Loose file under spaced filename, exact hash | 17 (6) | 6 counted; 4 with outputs | Executed | BAGLS Mean Shift/K-means and Dice/IoU |
| 7 | `01-Introduction_traitement_images_medicales.ipynb` / `9be69950` | Bundled exact hash | 36 (14) | none | Template/source | TP1 introduction |
| 8 | `02-Pretraitement_images_medicales.ipynb` / `af79a1a7` | Bundled exact hash | 36 (15) | none | Template/source | noise, filters, thresholds, morphology, contours, watershed |
| 9 | `03-Segmentation_images_medicales.ipynb` / `792d5da5` | Bundled exact hash | 13 (2) | none; both code cells empty | Empty template | segmentation headings/instructions only |
| 10 | `atelier2_part1_multi-modalites.ipynb` / `b000dd5f` | Bundled exact hash | 15 (7) | none | Template/source | endoscopy plus T1/T2 MRI filtering and thresholding |
| 11 | `complement_atelier1.ipynb` / `535c7cb4` | Bundled exact hash | 9 (3) | none; all three code cells empty | Empty template | explanatory supplement |

The manifest also reports `04_UNET_Segmentation.ipynb` as an exact duplicate of row 4 (same hash) and excludes it from the unique count. That duplicate is not currently present for independent checking.

## Duplicate artifact classification

Content hash, not filename, was used:

- `C:\Users\aziza\Downloads\03_Segmentation_Mme_ATTIA (1).ipynb` and `C:\Users\aziza\Downloads\03_Segmentation_Mme_ATTIA.ipynb` are byte-identical (`361e7dd5…`); one unique executed artifact.
- Loose `C:\Users\aziza\Downloads\03-Segmentation_Mme_ATTIA.ipynb` is byte-identical (`792d5da5…`) to bundled `reference_notebooks/03-Segmentation_images_medicales.ipynb`; it is the empty source template, not the missing executed row 2.
- The manifest-reported `04_UNET_Segmentation.ipynb` duplicates executed TP4 hash `74a86bf1…`; because the duplicate file and archive are absent, this classification is manifest-supported rather than independently rehashed in Phase 1B.
- Similar filenames with different hashes were not collapsed. In particular, hyphen/underscore variants of segmentation notebooks represent different artifacts.

## Saved-output findings

No inspected notebook contains a saved exception output.

- Executed TP1 contains stream and display outputs but defines no reusable functions.
- Executed TP3 row 6 records one BAGLS image resized from 128×288 to 114×256, plus historical Dice/IoU values for Mean Shift and K-means. These are one-image notebook outputs, not general performance estimates.
- Executed TP4 records 647 benign/malignant BUSI images, a 582/65 random split, historical 10- and 50-epoch training output, and saved metrics. These metrics were not reproduced. A compatible 50-epoch checkpoint was supplied later at `models/unet_breast_batch16_epochs50.hdf5`; it was not present in the audited notebook archive.
- Executed FIRE row 5 records discovery of 134 annotated pairs, 402 model evaluations, a synthetic self-test, and a benchmark table. The synthetic result is explicitly distinct from FIRE. The dataset and result CSVs are not supplied, so the saved table was not recomputed.
- Rows 7–11 have no execution counts and no outputs. Rows 9 and 11 contain no executable code at all.

Selected historical values, transcribed only to identify the saved evidence: TP3 row 6 reports Mean Shift Dice 0.8337/IoU 0.7148 and K-means Dice 0.8301/IoU 0.7095 on its single BAGLS example. TP4 reports, respectively for 10 and 50 epochs, validation loss 0.2075/0.1938, pixel accuracy 92.65%/95.22%, pooled IoU 0.2974/0.5520, and mean per-image Dice 0.3680/0.6927. The FIRE summary reports all 402 transformation fits with status `ok`, but its success definition is TRE below 50 pixels and is not equivalent to merely fitting a transform. None of these values was reproduced during this audit, and none should be presented as a current platform benchmark.

## Datasets and assets referenced

| Notebook family | Referenced data | Present in prototype? | Notes |
|---|---|---|---|
| TP1/TP2 | `frame856.png`, `frame442.png`, `frame442_glottis.png`, `patient4.mp4`, `01468_rgb.png`, `seg1468.png` | No | Relative `Data_At1`/`Data_At2` and Colab-style discovery |
| Executed TP3 | `bagls_test_2.png`, `bagls_test_2_mask.png` | Yes, under `assets/` | Bundled pair permits a small regression fixture, not a dataset benchmark |
| Multi-modal atelier | `frame442.png`, `Tr-pi_1.jpg`, `Te-no_264.jpg` | No | Endoscopy, MRI T1, MRI T2; despite some prose, no radiograph is supplied |
| TP4 | Google Drive `breast_dataset (1)/Dataset_BUSI_with_GT/{benign,malignant,normal}` | No | Normal cases excluded by `INCLUDE_NORMAL=False`; multiple masks merged by maximum |
| FIRE | FIRE images plus `control_points_*_1_2.txt` under Kaggle input | No | Saved output reports 134 pairs; source dataset is absent locally |

## Normalization and split rules

- TP1/TP2 and TP3 operate mainly on uint8 display images; `skimage.color.rgb2gray` produces floating grayscale for TP3 class selection.
- Multi-modal template `_to_uint8` leaves uint8 unchanged, scales values above 255 by the image maximum, and otherwise casts directly. This is lossy and not suitable as a universal medical-image normalization rule.
- TP4 resizes grayscale images to 256×256 with OpenCV default linear interpolation, then calls `tensorflow.keras.utils.normalize(image_dataset, axis=1)` on shape `(N,H,W)`, then adds a singleton channel. This is L2 normalization down the height axis for every batch item/column. Masks use nearest-neighbor resize, union via `np.maximum`, threshold `>127`, and float32 binary values.
- TP4 uses `train_test_split(..., test_size=0.10, random_state=42)` with no stratification, patient grouping, or separate test set visible in the code.
- FIRE uses the green channel with CLAHE for feature detection, resizes each image independently to maximum side 950 while tracking exact scale matrices, estimates transforms without landmarks, and evaluates landmarks in original FIRE pixel coordinates.

## Inventory uncertainties

The absent archive prevents proof that the loose notebooks are the intended updated input rather than coincidentally matching artifacts. Exact hashes give strong identity for nine rows. For rows 2 and 3, only manifest metadata is available; their exact code and saved outputs must not be quoted as independently inspected. A loose file named `03-Segmentation_Mme_ATTIA.ipynb` is present but hashes to row 9’s empty template, not to missing row 2—filename alone is therefore unsafe for identity.

Original preservation was rechecked through SHA-256 after testing; the nine available notebook hashes remain exactly those listed in the table. Tests and Streamlit used project Python modules and bundled PNG fixtures, not notebook execution.
