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

## Optuna tuning for biomolecules

The seven-class experiment uses adenine, dl-ala, dl-phe, dl-tyr, glu, RNA, and Lipid only. After the notebook's carbon-peak-cut cell, run the Optuna cell immediately below it. It takes the prepared `renishaw_everything` and `horiba_everything` frames directly; no earlier cells need to be rerun. It tunes on a stratified Renishaw training/validation split, then evaluates once on a separate Renishaw test split and all Horiba spectra. Confusion matrices, metrics, and weights are saved under `optuna_results/`.

For a standalone run that rebuilds the same prepared spectra from the Drive files, use:

```bash
.venv/bin/python optuna_biomolecules.py --trials 8 --epochs 8
```

For the adenine/dl-ala/RNA-only experiment, run its dedicated notebook cell or:

```bash
.venv/bin/python optuna_biomolecules.py --classes adenine dl-ala RNA --trials 8 --epochs 8
```

Its results are isolated in `optuna_results/adenine_dl-ala_RNA/`.

Rows from the same acquisition may be correlated, so the held-out Renishaw score measures within-dataset generalization, not necessarily new experimental batches. The Horiba score is a cross-instrument check.

## Reproducible spectral diagnostics

The final notebook section, **Reproducible biomolecule spectral diagnostics**, can run
independently in a fresh project kernel. Run only that section to avoid starting the
historical transformer/Optuna experiments. It reads `RAMAN_DATA_DIR` directly and uses
`spectral_diagnostics.py`; it does not reuse earlier notebook dataframes.

The analysis uses a seed of 42, fixed per-class row selection, a common 450–1450 cm⁻¹
grid, and three matched preprocessing views (as loaded, ASLS, ASLS + min–max). It saves
spectral means/spread, intensity and shape comparison matrices, lag/peak screens,
leave-one-molecule-out affine response predictions, and residual/information maps.
Plots are descriptive diagnostics, not classification accuracies or impossibility proofs.

The last cell reruns the analysis and asserts exact equality of selected rows, input
arrays, numerical outputs, metric tables, CSVs, and PNGs. Results and input hashes,
selected row IDs, configuration, code hash, and package versions are saved in
`spectral_diagnostics_results/<run_id>/` (ignored by Git). Numerical repeatability is
scoped to identical inputs, code, configuration, and numerical environment; seeding
alone cannot ensure bitwise results across different platforms or package versions.

Lipid/RNA labels inherit the notebook's row-order convention (375 RNA, then 375 Lipid).
Those CSVs may have upstream preprocessing; “as loaded” is not a claim of raw provenance.
No carbon cut, quality filtering, KNN pairing, GPU training, or target-label tuning is used.

Run the focused correctness checks with:

```bash
.venv/bin/python -m unittest test_spectral_diagnostics -v
```


## Original local RNA/lipid files

Use the notebook's final section **Original RNA/lipid files from Downloads — local
loading and diagnostics**. It runs independently in a fresh project kernel, loads all
original local RNA/lipid TXT files, and rebuilds the seven-molecule diagnostics with
those inputs. Files remain in Downloads; nothing is uploaded or copied to Drive.

`raw_lipid_rna_manifest.json` explicitly records eight files, molecule labels, inferred
instrument assignments, and SHA-256 checksums. `raw_lipid_rna.py` verifies each export's
structure and checksum, reconstructs complete spectra, preserves file/index/X/Y
provenance, and interpolates only within native coverage. No quality filtering,
normalization, or baseline subtraction occurs in the loader. Format-based instrument
assignments match the known project exports; vendor instrument IDs are not embedded.

The default raw-file root is `~/Downloads`. Set `RAMAN_RAW_DIR` in `.env` or the notebook
if the files move while preserving their subfolders. Missing/changed files fail loudly;
the raw-input route never falls back to the filtered Drive CSVs. Do not change checksums
without reviewing changed source files.

The common raw-input analysis grid is **451–1450 cm⁻¹**, because original RNA/lipid
exports start at 450.736/450.855 cm⁻¹. The historical 450 endpoint would require
extrapolation. Other biomolecules are evaluated on the same new grid. Seed 42 selects
750 per molecule/instrument for diagnostics, with recorded source rows. All 12,462
original RNA/lipid spectra are also available in the notebook's
`renishaw_lipid_rna_raw` / `horiba_lipid_rna_raw` frames, whose final three columns retain
the existing metadata convention. File/position provenance is kept separately to avoid
using it as a spectral feature.

Raw-input results are isolated under `spectral_diagnostics_results/raw_local/`.
Historical diagnostic and classifier results remain intact and use their prior input
cohorts; they are not automatically converted into raw-input training experiments.
The user confirmed on 2026-10-22 that the earlier lipid/RNA CSVs were preprocessed and
filtered, so old results should not be described as tests of the full raw cohort.

Python entry point:

```python
from pathlib import Path
from spectral_diagnostics import DiagnosticConfig, run_diagnostics

result = run_diagnostics(
    data_dir, output_dir,
    DiagnosticConfig(seed=42, low=451, high=1450),
    raw_lipid_rna_manifest=Path("raw_lipid_rna_manifest.json"),
    raw_downloads_dir=Path.home() / "Downloads",
)
```

Run loader and existing diagnostic checks with:

```bash
.venv/bin/python -m unittest test_raw_lipid_rna test_spectral_diagnostics -v
```
