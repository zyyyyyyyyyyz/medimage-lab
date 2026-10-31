# Recommended architecture

## Recommendation

Use an incremental “strangler” migration: keep the Streamlit prototype as a read-only behavior reference while extracting tested Python services behind a small FastAPI API, then build a React/Vite TypeScript client against an OpenAPI contract. Do not rewrite algorithms and UI simultaneously. Registration and U-Net should remain optional capabilities; the latter must report unavailable until a verified checkpoint is installed.

For a university project, a single deployable backend process plus static frontend is the best complexity trade-off. Do not introduce a database, message broker, Kubernetes, or distributed object storage for the first release. Add an asynchronous worker only if measured timeouts show that bounded synchronous jobs are inadequate.

```text
React/Vite UI
  upload + parameters + provenance banner + result viewer
                     │ HTTPS / OpenAPI
FastAPI application  │
  routers ─ schemas ─ services ─ algorithm adapters
       │          │             ├─ conventional (required)
       │          │             ├─ BUSI U-Net (optional capability)
       │          │             └─ FIRE registration (later optional)
       │          └─ validation / modality policy / metric policy
       └─ health + capability/model manifest
                     │
       bounded in-memory/temp workspace; delete at request end
```

## Backend layout

```text
backend/
  pyproject.toml
  app/
    main.py
    api/v1/{health,capabilities,analysis}.py
    schemas/{requests,responses,provenance}.py
    services/{analysis,metrics,model_registry}.py
    imaging/{io,validation,normalization}.py
    algorithms/{conventional,registration}.py
    models/busi_unet.py
    settings.py
  tests/{unit,integration,scientific}/
```

- Preserve pure NumPy arrays and functions below the service layer. No Streamlit/FastAPI imports in algorithm modules.
- Use Pydantic discriminated configurations for each method; do not accept an unvalidated dictionary of arbitrary parameters.
- Return a run manifest with method/version, parameter values, input dimensions/type, preprocessing identifier, duration, warnings, model checksum when applicable, and whether metrics had valid ground truth.
- Keep uploads stateless for MVP. Decode once into a per-request temporary directory or memory buffer, enforce byte/pixel/time limits, and delete on completion. Never log image bytes or patient metadata.
- Run CPU-bound algorithms in a bounded executor and cap concurrent expensive methods. A request timeout must yield a clear per-method error rather than taking down the process.
- Load a verified model once at startup/lazily behind a lock. Capability discovery should return `available=false` with a reason when TensorFlow or the checkpoint is absent.

Suggested MVP API:

| Method/path | Purpose |
|---|---|
| `GET /api/v1/health` | liveness only |
| `GET /api/v1/capabilities` | available algorithms, limits, U-Net status/model manifest |
| `POST /api/v1/analysis` | multipart image, optional aligned mask, declared modality, and JSON method configs; returns masks/metrics/provenance |

For a small image, masks can be base64 PNGs in the response; a multipart response or short-lived artifact endpoint is preferable if payload size becomes material. Do not persist uploads by default. Reject mismatched reference dimensions in the scientific API unless the request includes a named, explicit transform.

## Frontend layout

```text
frontend/
  src/
    app/
    api/generated/
    features/{upload,configuration,results,comparison}/
    components/
    scientific/{warnings,provenance}/
    test/
  package.json
  vite.config.ts
```

Use React + TypeScript + Vite, a generated OpenAPI client, and simple local state initially. Keep original, processed image, masks, overlays, and metric provenance visibly distinct. Disable incompatible choices based on server capabilities and modality; do not rely on a warning after submission. Label the tool educational/non-diagnostic and show when results are unvalidated or a model is unavailable.

## Relationship to Streamlit

Streamlit remains runnable as a reference until parity tests pass. First make it call the same extracted service functions locally; then use it as a comparison harness while React is built. Retire it only after the React workflow satisfies the acceptance tests. This reduces the risk of silently changing algorithm parameters during UI migration.

## Security and medical-data posture

- Educational research tool, not a medical device and not for diagnosis.
- Initially allow only anonymized, non-sensitive demo/research images. Do not claim DICOM support until metadata de-identification, transfer-syntax handling, orientation, spacing, and pixel transformations are designed and tested.
- Enforce content sniffing, allowed dimensions/channels, compressed and decoded size limits, request timeouts, concurrency limits, restrictive CORS, and safe error messages.
- If multi-user deployment is later required, add authentication, retention/deletion policy, access logging without PHI, encrypted transport/storage, threat modeling, and institutional/legal review.

## Complexity choices

| Choice | MVP | Defer until evidence requires it |
|---|---|---|
| Processing | synchronous, bounded FastAPI executor | queue/worker for long registration or batch jobs |
| Storage | request-scoped temporary files | database/object store/session history |
| Deployment | one backend container + static frontend | orchestration/auto-scaling |
| Models | registry interface, BUSI disabled without verified weights | multiple modality-specific models |
| Formats | PNG/JPEG/TIFF demo rasters | DICOM/NIfTI and volumetric processing |
| Registration | separate optional module | expose after extraction and benchmark fixtures |
