# Verified development environment

Verified on 2026-10-31 with Windows and Python 3.11.9. The environment is project-local at `.venv`; no global package was changed. TensorFlow was absent from the conventional baseline and was later installed from `requirements-dl.txt` for the supplied TP4 checkpoint.

## Reproduce from the lock

From the project root in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip==24.3.1
.\.venv\Scripts\python.exe -m pip install -r requirements-baseline-lock.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -vv
```

If `.venv` already exists, do not mix it with a different Python installation. Remove or rename that project-local directory outside this procedure, then recreate it. Never run these commands against the global Python by omitting `.\.venv\Scripts\python.exe` after creation.

## Why this baseline

The original global environment had NumPy 2.4.6 with scikit-learn 1.2.2 and pandas 1.5.3. Both compiled extensions failed to import with `numpy.dtype size changed`. The clean environment uses the verified compatible core set:

| Package | Version |
|---|---:|
| Python | 3.11.9 |
| pip | 24.3.1 |
| NumPy | 2.1.3 |
| SciPy | 1.14.1 |
| scikit-learn | 1.5.2 |
| scikit-image | 0.24.0 |
| pandas | 2.2.3 |
| OpenCV headless | 4.10.0.84 |
| Pillow | 11.0.0 |
| Streamlit | 1.40.2 |
| pytest | 8.3.4 |

All direct and transitive versions observed in the passing environment are frozen in `requirements-baseline-lock.txt`. `requirements.txt` remains the project’s supported version-range declaration; the lock is the reproducible Windows/Python 3.11 baseline.

## Verification commands and outcomes

```powershell
.\.venv\Scripts\python.exe --version
# Python 3.11.9

.\.venv\Scripts\python.exe -m pip check
# No broken requirements found.

.\.venv\Scripts\python.exe -c "import cv2,numpy,pandas,scipy,skimage,sklearn,streamlit; print('imports=OK')"
# imports=OK
```

To run the application:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Optional deep-learning environment

Do not install `requirements-dl.txt` for conventional development. TensorFlow remains an optional extra because no verified U-Net checkpoint is supplied. If the authentic checkpoint is later recovered, create and validate a separate compatible environment or explicitly install:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dl.txt
```

Installing TensorFlow does not make the model pretrained. Inference remains unavailable until `models/unet_breast_batch16_epochs50.hdf5` is supplied with verified provenance and checksum.

## Updating the lock

Update only in a disposable clean environment, run the complete test and smoke-test sequence, then capture exact packages with:

```powershell
.\.venv\Scripts\python.exe -m pip freeze --all
```

Do not regenerate the lock from the previously broken global environment.
