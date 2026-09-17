# FMKCL Raman transformer notebook

This repository is configured to run `fmkcl_transformer.ipynb` locally in VS Code. It no longer depends on Google Colab or a mounted Google Drive.

## 1. Create the Python environment

Python 3.11 is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Open this folder in VS Code, open the notebook, and select `.venv/bin/python` as the kernel.

## 2. Supply the data

By default, the notebook expects all source files directly inside `data/`. That folder is ignored by Git because the spectra may be large or private.

If the files live elsewhere, copy `.env.example` to `.env` and update `RAMAN_DATA_DIR`:

```bash
cp .env.example .env
```

Alternatively, export the variable before launching VS Code:

```bash
export RAMAN_DATA_DIR="/absolute/path/to/Virus Stuff"
code .
```

The data setup cell validates every expected filename and reports all missing files at once. Trained model weights are written to `models/`.

## Notes

- CUDA is used when available, Apple Metal (`mps`) is used on supported Macs, and CPU is the fallback.
- Dependency installation is kept outside the notebook to avoid mutating or desynchronizing its active kernel.
- Run cells from top to bottom after selecting the project kernel.
