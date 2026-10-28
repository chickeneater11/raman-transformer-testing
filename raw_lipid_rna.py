"""Load original local RNA/lipid exports without uploading or preprocessing them.

Instrument assignments are high-confidence inferences from export formats matching
the project's known instrument files. They are not vendor-embedded identification.
"""
from hashlib import sha256
from pathlib import Path
import json
import os

import numpy as np
import pandas as pd


INSTRUMENTS = ("Renishaw", "Horiba")
MOLECULES = ("RNA", "Lipid")


def file_hash(path):
    h = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def detect_export_format(path):
    """Recognize structure rather than relying on an instrument-like filename."""
    with Path(path).open(encoding="utf-8-sig") as handle:
        first = next((line.rstrip("\r\n") for line in handle if line.strip()), "")
    fields = first.split("\t")
    if len(fields) == 4:
        try:
            numeric = np.asarray(fields, dtype=float)
        except ValueError:
            pass
        else:
            if np.isfinite(numeric).all():
                return "renishaw_long_xy_wavenumber_intensity"
    if len(fields) > 4 and fields[:2] == ["", ""]:
        try:
            wn = np.asarray(fields[2:], dtype=float)
        except ValueError:
            pass
        else:
            if np.isfinite(wn).all() and np.all(np.diff(wn) > 0):
                return "horiba_wide_xy_spectral_header"
    raise ValueError(f"Unrecognized Raman export structure: {path}")


def read_raw_export(path, expected_instrument=None):
    path = Path(path)
    layout = detect_export_format(path)
    inferred = "Renishaw" if layout.startswith("renishaw") else "Horiba"
    if expected_instrument is not None and expected_instrument != inferred:
        raise ValueError(f"{path.name}: manifest assigns {expected_instrument}, but detected {inferred}-style format")
    if inferred == "Horiba":
        frame = pd.read_csv(path, sep="\t", encoding="utf-8-sig")
        wn = np.asarray(frame.columns[2:], dtype=np.float64)
        xy = frame.iloc[:, :2].to_numpy(dtype=np.float64)
        spectra = frame.iloc[:, 2:].to_numpy(dtype=np.float64)
    else:
        # One export is held at a time; reshape complete contiguous spectra instead
        # of constructing a multi-million-row pivot table.
        values = pd.read_csv(path, sep="\t", header=None, dtype=np.float64).to_numpy()
        if values.shape[1] != 4 or not len(values) or not np.isfinite(values).all():
            raise ValueError(f"{path.name}: expected finite x, y, wavenumber, intensity columns")
        starts = np.flatnonzero(values[:, 2] == values[0, 2])
        lengths = np.diff(np.r_[starts, len(values)])
        if not len(starts) or len(np.unique(lengths)) != 1:
            raise ValueError(f"{path.name}: incomplete or unequal spectrum blocks")
        points = int(lengths[0])
        if points < 2:
            raise ValueError(f"{path.name}: spectrum must contain multiple wavenumbers")
        blocks = values.reshape(len(starts), points, 4)
        wn = blocks[0, :, 2].copy()
        if not np.all(blocks[:, :, 2] == wn):
            raise ValueError(f"{path.name}: spectrum grids differ within file")
        xy = blocks[:, 0, :2].copy()
        if not np.all(blocks[:, :, :2] == xy[:, None, :]):
            raise ValueError(f"{path.name}: positions change inside a spectrum block")
        spectra = blocks[:, :, 3].copy()
    if not len(spectra) or not np.isfinite(spectra).all() or not np.isfinite(xy).all():
        raise ValueError(f"{path.name}: empty or nonfinite spectral/coordinate data")
    order = np.argsort(wn, kind="stable")
    wn, spectra = wn[order], spectra[:, order]
    if not np.isfinite(wn).all() or np.any(np.diff(wn) <= 0):
        raise ValueError(f"{path.name}: invalid or duplicate spectral coordinates")
    return {"wavenumbers": wn, "spectra": spectra, "xy": xy,
            "instrument": inferred, "format": layout}


def interpolate_without_extrapolation(wn, spectra, grid, name):
    grid = np.asarray(grid, dtype=np.float64)
    if grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0):
        raise ValueError("Target grid must be finite, strictly increasing, and nonempty")
    if grid[0] < wn[0] or grid[-1] > wn[-1]:
        raise ValueError(f"{name}: target grid {grid[0]:g}–{grid[-1]:g} exceeds native {wn[0]:g}–{wn[-1]:g}; refusing extrapolation")
    return np.stack([np.interp(grid, wn, row) for row in spectra])


def load_raw_lipid_rna(manifest_path, grid, downloads_dir=None, max_per_class=None, seed=42):
    """Return aligned intensity arrays and explicit per-spectrum provenance.

    Only coordinate sorting and interpolation occur here: no baseline subtraction,
    intensity normalization, quality filtering, or cross-instrument spectrum pairing.
    Optional selection uses independent local PCG64 streams, with sorted source IDs.
    """
    manifest_path = Path(manifest_path).resolve()
    if max_per_class is not None and max_per_class < 1:
        raise ValueError("max_per_class must be positive or None")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    root = Path(downloads_dir or os.environ.get("RAMAN_RAW_DIR") or Path.home() / "Downloads").expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported raw manifest schema")
    entries = sorted(manifest["files"], key=lambda f: (INSTRUMENTS.index(f["instrument"]), MOLECULES.index(f["molecule"]), f["relative_path"]))
    chunks = {(i, m): [] for i in INSTRUMENTS for m in MOLECULES}
    provenance = {key: [] for key in chunks}
    inventories, inputs, seen_hashes, seen_paths = [], [], set(), set()
    for entry in entries:
        relative = Path(entry["relative_path"])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError(f"Raw manifest path must be relative to RAMAN_RAW_DIR: {relative}")
        if path in seen_paths:
            raise ValueError(f"Duplicate file path in raw manifest: {relative}")
        seen_paths.add(path)
        if not path.is_file():
            raise FileNotFoundError(f"Missing local raw file: {path}. Set RAMAN_RAW_DIR if it moved; no legacy CSV fallback is performed.")
        digest = file_hash(path)
        if entry.get("sha256") and digest != entry["sha256"]:
            raise ValueError(f"{relative}: file content changed since the recorded inventory; review and update the manifest")
        if digest in seen_hashes:
            raise ValueError(f"Duplicate file content would double-count spectra: {relative}")
        seen_hashes.add(digest)
        parsed = read_raw_export(path, entry["instrument"])
        wn, raw = parsed["wavenumbers"], parsed["spectra"]
        if file_hash(path) != digest:
            raise ValueError(f"{relative}: file changed while it was being read")
        key = (entry["instrument"], entry["molecule"])
        chunks[key].append(interpolate_without_extrapolation(wn, raw, grid, relative))
        provenance[key].append(pd.DataFrame({
            "instrument": key[0], "class": key[1], "file": relative.as_posix(),
            "source_row_zero_based": np.arange(len(raw)),
            "source_id": [f"{relative.as_posix()}::spectrum:{j}" for j in range(len(raw))],
            "x": parsed["xy"][:, 0], "y": parsed["xy"][:, 1],
        }))
        inventories.append({"instrument": key[0], "class": key[1], "file": relative.as_posix(),
                            "spectra": len(raw), "native_points": len(wn), "native_min": float(wn[0]),
                            "native_max": float(wn[-1]), "bytes": path.stat().st_size,
                            "sha256": digest, "detected_format": parsed["format"],
                            "assignment_basis": "Format matches known project instrument exports; instrument ID is inferred"})
        inputs.append({"name": relative.as_posix(), "sha256": digest, "bytes": path.stat().st_size})
    data, records, counts = {}, [], []
    for key in chunks:
        if not chunks[key]:
            raise ValueError(f"Raw manifest has no files for {key}")
        values = np.concatenate(chunks[key])
        ids = np.arange(len(values))
        if max_per_class is not None and max_per_class < len(values):
            key_seed = int.from_bytes(sha256(f"{key[0]}/{key[1]}".encode()).digest()[:8], "little")
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, key_seed])))
            ids = np.sort(rng.choice(len(values), size=max_per_class, replace=False))
        data[key] = values[ids]
        records.append(pd.concat(provenance[key], ignore_index=True).iloc[ids].reset_index(drop=True))
        counts.append({"instrument": key[0], "class": key[1], "available": len(values), "selected": len(ids)})
    return {"data": data, "grid": np.asarray(grid, dtype=np.float64),
            "rows": pd.concat(records, ignore_index=True), "inventory": pd.DataFrame(inventories),
            "counts": pd.DataFrame(counts), "inputs": inputs, "manifest_path": manifest_path,
            "manifest_sha256": file_hash(manifest_path), "loader_sha256": file_hash(__file__),
            "data_source": "original_local_txt"}


def as_notebook_frames(loaded):
    """Frames keep the notebook's spectral-columns + three metadata convention.

    Full file/coordinate provenance remains in loaded['rows']; it is not accidentally
    included as an input feature by existing positional spectral-column slicing.
    """
    frames = {}
    for instrument in INSTRUMENTS:
        parts = []
        for label in MOLECULES:
            frame = pd.DataFrame(loaded["data"][instrument, label], columns=loaded["grid"])
            frame["Virus"], frame["Subtype"], frame["Strain"] = label, None, None
            parts.append(frame)
        frames[instrument] = pd.concat(parts, ignore_index=True)
    return frames
