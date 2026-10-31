# Baseline test report

Execution date: 2026-10-31. Platform: Windows, Python 3.11.9, project-local `.venv`. TensorFlow was not installed and no U-Net checkpoint was present.

## Dependency validation

Command:

```powershell
.\.venv\Scripts\python.exe -m pip check
```

Actual result: `No broken requirements found.` Imports of OpenCV, NumPy, pandas, SciPy, scikit-image, scikit-learn, and Streamlit all succeeded.

## Automated tests

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest -vv
```

Actual result:

```text
collected 4 items
tests/test_algorithms.py::test_scores_exact PASSED
tests/test_algorithms.py::test_methods_shapes PASSED
tests/test_algorithms.py::test_preprocessing_and_overlay PASSED
tests/test_algorithms.py::test_tp4_normalization_equivalence PASSED
4 passed, 1 warning in 3.22s
```

An earlier verbose run immediately after installation also passed all four tests in 6.49 seconds. The 3.22-second value above is the final verification run after documentation and lock creation; timing variation does not change the outcomes.

The warning is a scikit-learn `ConvergenceWarning`: the synthetic test image has only two distinct intensity values while K-means requests three clusters. This does not invalidate the asserted mask shape/binary-output contract, and it was not hidden or suppressed. No production or test code change was required.

Coverage represented by the four tests:

- exact Dice/IoU identity, disjoint, and empty behavior exercised by `test_scores_exact`;
- output dimensions and binary values for Otsu, manual threshold, K-means, Mean Shift, Chan–Vese, and watershed;
- shape preservation for all prototype preprocessors plus overlay and mask cleaning;
- numerical equivalence of TP4 resize plus axis-1 L2 normalization and channel expansion.

Not covered: real-dataset scientific accuracy, registration, uploaded-file validation, browser interaction, TensorFlow model construction, checkpoint loading, or U-Net inference.

## Streamlit smoke tests

Script execution used Streamlit’s test harness:

```powershell
.\.venv\Scripts\python.exe -c "from streamlit.testing.v1 import AppTest; at=AppTest.from_file('app.py', default_timeout=60).run(); ..."
```

Actual result:

```text
exceptions 0
titles ['MedImage Lab']
dataframes 1
warnings 0
errors 0
```

The harness emitted the standard bare-mode “missing ScriptRunContext” warning before execution; the application itself reported no exception or error.

A real headless server was then launched on port 8765:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.headless=true --server.port=8765 --browser.gatherUsageStats=false
```

`Invoke-WebRequest http://127.0.0.1:8765/` returned HTTP 200, `text/html`, 891 bytes, with the Streamlit HTML title. The server was then deliberately stopped with Ctrl+C; its resulting interrupt exit code is not an application startup failure.

## Model capability

TensorFlow discovery returned false. A checkpoint scan excluding `.venv` found no `.h5`, `.hdf5`, `.keras`, `.pt`, `.pth`, `.onnx`, `.ckpt`, or `.safetensors` project asset. U-Net inference was therefore not executed and is not claimed to work with trained weights. The architecture and preprocessing test are not evidence of a pretrained model.

## Baseline conclusion

The conventional-processing development baseline is operational and reproducible from the lock. The application starts and the existing suite passes. Scientific validation and trained U-Net inference remain outside this baseline.

## Final implementation extension

After the baseline was accepted, three targeted end-to-end tests were added for upload validation, all preprocessing paths, all six conventional BAGLS segmentations, aligned metrics, PNG/CSV exports, and rejection of misaligned ground truth. That intermediate suite collected 7 tests and passed all 7 in 11.77 seconds without warnings.

The later strict-fidelity correction added four parity tests and rerouted TP2/TP3 through faithful modules. That pre-checkpoint suite collected 11 tests; `TP_PARITY_REPORT.md` is authoritative for current parity classifications and evidence.

## TP4 checkpoint integration update

After the conventional baseline, the supplied TP4 checkpoint was integrated with TensorFlow 2.20.0 in the same project-local `.venv`. The final suite now collects 12 tests and passes all 12 in 8.08 seconds. A real `load_model(..., compile=False)` and CPU prediction succeeded. This later result does not retroactively change the historical baseline evidence above, and it does not reproduce any notebook accuracy metric.
