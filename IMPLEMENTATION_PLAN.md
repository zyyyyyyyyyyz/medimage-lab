# Phased implementation plan

Phase 1B baseline work was performed on 2026-10-31 without starting the FastAPI/React migration.

## Phase 1B status

| Item | Status | Evidence |
|---|---|---|
| Project-local Python 3.11 environment | Complete | `.venv`, `ENVIRONMENT_SETUP.md` |
| Exact conventional dependency lock | Complete | `requirements-baseline-lock.txt`; `pip check` clean |
| Four existing tests | Complete | 4 collected, 4 passed; `BASELINE_TEST_REPORT.md` |
| Streamlit startup | Complete | test harness zero exceptions; HTTP 200 headless smoke test |
| TensorFlow/checkpoint | Integrated | TensorFlow 2.20.0 in project `.venv`; supplied TP4 checkpoint loads and predicts |
| Eleven notebook identities accounted for | Partial | nine direct JSON/hash verifications; two manifest-only identities |
| Requested archive extraction | **Blocked** | `Medical_TP_Notebooks_Complete_Updated.zip` absent from project root/tree and immediate parent |

Readiness decision: the conventional baseline is ready, but Phase 2 is not authorized by the stated acceptance conditions until the archive is supplied and verified, or the owner explicitly accepts the two missing originals as unavailable.

## Phase 0 — recover and freeze evidence

Work: obtain `Medical_TP_Notebooks_Complete_Updated.zip`; verify its checksum and reconcile the two missing manifest artifacts; record dataset/checkpoint licenses and hashes; create a clean supported Python environment and lock dependencies; decide whether U-Net remains disabled or a genuine checkpoint will be supplied.

Acceptance criteria:

- all 11 unique notebooks are available by exact hash, or missing rows are explicitly accepted as unavailable;
- originals live in a read-only/reference location and hash verification is automated;
- a clean install from the lock succeeds on the chosen Python version;
- existing pytest collection succeeds; every result is reported without suppressing warnings/errors;
- checkpoint capability reports unavailable unless the expected file passes format, input-shape, and checksum/provenance validation.

Tests: manifest/hash test; clean-container install; import smoke test; `pytest`; checkpoint absent/wrong-file/wrong-shape tests.

Current completion: environment lock, imports, pytest, application startup, and checkpoint-absence checks are complete. Archive recovery, direct JSON verification of rows 2 and 3, automated archive hash inventory, and dataset/checkpoint provenance remain open.

## Phase 1 — stabilize the scientific core

Work: separate pure I/O, preprocessing, segmentation, metrics, overlays, and U-Net adapters from Streamlit. Decide and document notebook-parity versus prototype-adapted behavior for every method. Add typed configurations and provenance records. Do not change algorithm parameters without a named version.

Acceptance criteria:

- pure functions have no UI/global state and validate dtype, shape, finite values, and binary-mask policy;
- every exposed algorithm has a source classification: exact, adapted, or new;
- TP4 axis-1 normalization matches a frozen reference array within tolerance;
- Dice/IoU empty-mask behavior and macro/micro definitions are explicit;
- mismatched masks are rejected unless a tested explicit transform is provided;
- BAGLS fixture outputs are deterministic for fixed seeds and OpenCV/sklearn versions.

Tests: unit/property tests for shapes, dtypes, constant images, tiny images, invalid clusters/kernels, deterministic seeds, metric edge cases, component filtering, interpolation, and normalization; golden-mask regression tests with tolerance/version metadata.

## Phase 2 — FastAPI backend

Work: add health, capabilities, and analysis endpoints; bounded resource handling; structured errors; model registry; request-scoped cleanup; OpenAPI schema.

Acceptance criteria:

- valid demo analysis returns binary masks, method parameters, timing, warnings, and metric provenance;
- malformed/oversized/unsupported inputs return documented 4xx responses;
- incompatible BUSI inference is rejected server-side;
- missing TensorFlow/checkpoint returns an unavailable capability, never random inference;
- temporary data is deleted on success and failure;
- concurrent expensive requests respect configured limits and timeouts.

Tests: FastAPI integration tests for all response paths; MIME/content mismatch; decompression/pixel limits; cleanup; concurrency; optional dependency absence; OpenAPI snapshot/contract test.

## Phase 3 — React/Vite frontend with Streamlit parity

Work: build typed upload/configuration/results flow; generate API client; display image, masks, overlays, provenance, metrics, and scientific warnings; preserve Streamlit until parity is proven.

Acceptance criteria:

- a user can reproduce every supported Streamlit MVP flow against the API;
- unavailable/incompatible models are disabled with a precise reason;
- metrics never appear without accepted aligned ground truth;
- exports include image/mask plus a machine-readable run manifest;
- loading, timeout, partial-method failure, and retry states are accessible and understandable.

Tests: Vitest/React Testing Library component tests; generated-client contract test; Playwright happy path and failure paths; visual regression for overlay/mask distinction; accessibility checks.

## Phase 4 — verified U-Net activation (conditional)

Prerequisite: user supplies the authentic checkpoint and permits its use. Do not train or download large weights as part of this phase without separate permission.

Acceptance criteria:

- checkpoint SHA-256, origin, license, training dataset/split, architecture, TensorFlow version, and preprocessing ID are recorded;
- model input/output shapes and finite prediction range pass startup validation;
- inference preprocessing is byte-for-byte/numerically consistent with the documented TP4 rule;
- regression predictions on approved BUSI fixtures match a frozen baseline within tolerance;
- evaluation uses a fixed, leakage-controlled BUSI test set and reports per-image Dice/IoU distributions, empty-mask policy, confidence intervals where appropriate, and hardware/runtime;
- no claim extrapolates to BAGLS, FIRE, MRI, or other ultrasound datasets.

Tests: corrupted/wrong model; checksum mismatch; model shape; preprocessing golden test; deterministic inference tolerance; batch/single consistency; approved BUSI benchmark script.

## Phase 5 — registration and medical formats (optional, separate scope)

Extract FIRE only after the segmentation MVP is stable. DICOM/NIfTI requires a separate design review.

Registration acceptance: landmarks remain evaluation-only; transforms are estimated from image features; TRE is in original pixels; overlap coverage and failures are retained in summaries; synthetic and FIRE results are labeled separately; results reproduce from a pinned dataset/version.

Medical-format acceptance: pixel spacing/orientation/rescale/window semantics are preserved; anonymization behavior is tested; original and display arrays are distinct; multi-frame/volume limits exist. Until then, the UI must say “display raster,” not imply full medical-format support.

## Release gate

A release is acceptable only when clean-environment backend and frontend suites pass, scientific golden tests pass, capability/model provenance is visible, no missing asset is represented as available, and the limitations in `AUDIT.md` are reflected in user-facing documentation. Any reported notebook number must be labeled historical until independently reproduced.
