"""Deterministic, classifier-free biomolecule spectral diagnostics.

No dependence on mutable notebook frames, global RNG state, Torch, or GPU kernels.
Repeatability is verified for identical inputs/config/code and numerical environment.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import json
import platform
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pybaselines.whittaker import asls
from scipy.signal import find_peaks, peak_widths, savgol_filter
from threadpoolctl import threadpool_limits


CLASSES = ("adenine", "dl-ala", "dl-phe", "dl-tyr", "glu", "RNA", "Lipid")
INSTRUMENTS = ("Renishaw", "Horiba")
VIEWS = ("as_loaded", "asls", "asls_minmax")
COLORS = ("#236A87", "#B65B29")


@dataclass(frozen=True)
class DiagnosticConfig:
    seed: int = 42
    max_per_class: int = 750
    low: int = 450
    high: int = 1450
    asls_lambda: float = 1e5
    asls_p: float = 0.01
    asls_max_iter: int = 50
    asls_tol: float = 1e-3
    coefficient_window: int = 31
    coefficient_polyorder: int = 2
    ridge_fraction: float = 1e-6
    low_information_ratio: float = 0.25
    max_lag_cm: int = 10
    rna_rows: int = 375
    expected_lipid_rna_rows: int = 750
    scatter_wavenumbers: tuple = (600, 730, 850, 1000, 1200, 1350)


def file_hash(path):
    h = sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_hash(array):
    a = np.ascontiguousarray(array)
    return sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def stable_indices(n, size, seed, key):
    # Each instrument/class has an independent stream; execution order cannot alter it.
    key_seed = int.from_bytes(sha256(key.encode()).digest()[:8], "little")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, key_seed])))
    return np.sort(rng.choice(n, size=min(n, size), replace=False))


def interpolate_checked(wn, spectra, grid, name):
    wn, spectra = np.asarray(wn, dtype=np.float64), np.asarray(spectra, dtype=np.float64)
    if spectra.ndim != 2 or len(wn) != spectra.shape[1] or not len(spectra):
        raise ValueError(f"{name}: invalid spectrum dimensions")
    if not np.isfinite(wn).all() or not np.isfinite(spectra).all():
        raise ValueError(f"{name}: nonfinite values; no silent row deletion is allowed")
    order = np.argsort(wn, kind="stable")
    wn, spectra = wn[order], spectra[:, order]
    if np.any(np.diff(wn) <= 0):
        raise ValueError(f"{name}: duplicate spectral coordinates")
    if wn[0] > grid[0] or wn[-1] < grid[-1]:
        raise ValueError(f"{name}: source grid does not cover requested grid; refusing extrapolation")
    if np.array_equal(wn, grid):
        return spectra.copy()
    return np.stack([np.interp(grid, wn, row) for row in spectra])


def read_amino_file(path, instrument):
    if instrument == "Horiba":
        frame = pd.read_csv(path, sep="\t").iloc[:, 2:]
        return frame.columns.astype(float).to_numpy(), frame.to_numpy(dtype=np.float64)
    # Preserve the notebook's spectrum reconstruction convention, without KNN pairing.
    frame = pd.read_csv(path, sep="\t", header=None).iloc[:, 2:]
    if frame.shape[1] != 2:
        raise ValueError(f"{path.name}: expected two position and two spectral columns")
    frame.columns = ["wavenumber", "intensity"]
    counts = frame.groupby("wavenumber", sort=True).size()
    if counts.nunique() != 1:
        raise ValueError(f"{path.name}: incomplete spectra in long-form data")
    frame["spectrum_index"] = frame.groupby("wavenumber", sort=False).cumcount()
    pivot = frame.pivot(index="spectrum_index", columns="wavenumber", values="intensity")
    return pivot.columns.to_numpy(dtype=float), pivot.to_numpy(dtype=np.float64)


def load_inputs(data_dir, cfg):
    grid = np.arange(cfg.low, cfg.high + 1, dtype=np.float64)
    data, rows, counts, inputs = {}, [], [], []
    for instrument in INSTRUMENTS:
        short = instrument.lower()
        blocks = {}
        for label in CLASSES[:5]:
            path = data_dir / f"{label}_{short}.txt"
            inputs.append({"name": path.name, "sha256": file_hash(path), "bytes": path.stat().st_size})
            wn, raw = read_amino_file(path, instrument)
            blocks[label] = (path, wn, raw, np.arange(len(raw)))
        path = data_dir / f"{short}_lipid_rna.csv"
        inputs.append({"name": path.name, "sha256": file_hash(path), "bytes": path.stat().st_size})
        frame = pd.read_csv(path)
        if len(frame) != cfg.expected_lipid_rna_rows:
            raise ValueError(f"{path.name}: expected {cfg.expected_lipid_rna_rows} rows for the notebook's positional RNA/Lipid labels; found {len(frame)}")
        wn, raw = frame.columns.astype(float).to_numpy(), frame.to_numpy(dtype=np.float64)
        for label, ids in (("RNA", np.arange(cfg.rna_rows)), ("Lipid", np.arange(cfg.rna_rows, len(raw)))):
            blocks[label] = (path, wn, raw[ids], ids)
        for label in CLASSES:
            path, wn, raw, source_ids = blocks[label]
            ids = stable_indices(len(raw), cfg.max_per_class, cfg.seed, f"{instrument}/{label}")
            if len(ids) < 2:
                raise ValueError(f"{instrument}/{label}: need at least two spectra")
            # Validate all source rows, so subsampling cannot conceal invalid data.
            if not np.isfinite(raw).all():
                raise ValueError(f"{path.name}: source contains nonfinite data")
            data[instrument, label] = interpolate_checked(wn, raw[ids], grid, path.name)
            counts.append({"instrument": instrument, "class": label, "available": len(raw), "selected": len(ids),
                           "native_points": len(wn), "native_min": float(np.min(wn)), "native_max": float(np.max(wn))})
            rows.extend({"instrument": instrument, "class": label, "file": path.name, "source_row_zero_based": int(i)} for i in source_ids[ids])
    return grid, data, pd.DataFrame(rows), pd.DataFrame(counts), inputs


def preprocess(data, cfg):
    views = {name: {} for name in VIEWS}
    audit = []
    for key, raw in data.items():
        corrected, nonconverged = [], 0
        for row in raw:
            baseline, params = asls(row, lam=cfg.asls_lambda, p=cfg.asls_p,
                                    max_iter=cfg.asls_max_iter, tol=cfg.asls_tol)
            corrected.append(row - baseline)
            nonconverged += int(params["tol_history"][-1] > cfg.asls_tol)
        corrected = np.stack(corrected)
        span = np.ptp(corrected, axis=1, keepdims=True)
        if not np.isfinite(corrected).all() or np.any(span <= 0):
            raise ValueError(f"{key}: invalid/constant baseline-corrected spectra")
        views["as_loaded"][key] = raw
        views["asls"][key] = corrected
        views["asls_minmax"][key] = (corrected - corrected.min(axis=1, keepdims=True)) / span
        audit.append({"instrument": key[0], "class": key[1], "asls_not_converged": nonconverged,
                      "selected": len(raw), "quality_rejections": 0})
    return views, pd.DataFrame(audit)


def moments(data):
    means = {i: np.stack([data[i, c].mean(axis=0) for c in CLASSES]) for i in INSTRUMENTS}
    sds = {i: np.stack([data[i, c].std(axis=0, ddof=1) for c in CLASSES]) for i in INSTRUMENTS}
    return means, sds


def correlation_matrix(a, b):
    ac, bc = a - a.mean(axis=1, keepdims=True), b - b.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(ac, axis=1)[:, None] * np.linalg.norm(bc, axis=1)[None, :]
    return np.divide(ac @ bc.T, denominator, out=np.full(denominator.shape, np.nan), where=denominator > 0).clip(-1, 1)


def rmse_matrix(a, b):
    return np.sqrt(np.mean((a[:, None, :] - b[None, :, :]) ** 2, axis=2))


def fit_shared_response(x, y, source_sd, cfg):
    """Equal molecule weights; ridge and smoothing use training molecules only.

    This is smoothed wavelength-wise affine regression, NOT a claim of physical
    instrument identification. Smoothness is fixed, never selected on held-out data.
    """
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - xm, y - ym
    variance = np.mean(xc ** 2, axis=0)
    scale = max(float(np.mean(variance)), np.finfo(float).tiny)
    gain = np.mean(xc * yc, axis=0) / (variance + cfg.ridge_fraction * scale)
    offset = ym - gain * xm
    gain = savgol_filter(gain, cfg.coefficient_window, cfg.coefficient_polyorder, mode="interp")
    offset = savgol_filter(offset, cfg.coefficient_window, cfg.coefficient_polyorder, mode="interp")
    spread = np.sqrt(np.mean(source_sd ** 2, axis=0))
    low_info = np.sqrt(variance) <= cfg.low_information_ratio * spread
    low_info |= variance <= 1e-12 * scale
    return offset, gain, low_info


def leave_one_molecule_out(means, sds, cfg):
    x, y = means["Renishaw"], means["Horiba"]
    predictions, gains, offsets, flags, outside = [], [], [], [], []
    records = []
    for held, label in enumerate(CLASSES):
        train = np.arange(len(CLASSES)) != held
        a, b, low_info = fit_shared_response(x[train], y[train], sds["Renishaw"][train], cfg)
        predicted = a + b * x[held]
        residual = predicted - y[held]
        spread = sds["Horiba"][held]
        normalized = np.divide(residual, spread, out=np.full_like(residual, np.nan), where=spread > 0)
        extrapolation = (x[held] < x[train].min(axis=0)) | (x[held] > x[train].max(axis=0))
        base_rmse = float(np.sqrt(np.mean((x[held] - y[held]) ** 2)))
        fitted_rmse = float(np.sqrt(np.mean(residual ** 2)))
        records.append({"class": label, "uncorrected_rmse": base_rmse, "held_out_rmse": fitted_rmse,
                        "rmse_change_pct": 100 * (fitted_rmse / base_rmse - 1) if base_rmse else np.nan,
                        "uncorrected_correlation": float(correlation_matrix(x[held:held+1], y[held:held+1])[0, 0]),
                        "held_out_correlation": float(correlation_matrix(predicted[None, :], y[held:held+1])[0, 0]),
                        "median_abs_residual_target_sd": float(np.nanmedian(np.abs(normalized))),
                        "low_information_fraction": float(low_info.mean()),
                        "extrapolation_fraction": float(extrapolation.mean()),
                        "negative_gain_fraction": float((b < 0).mean())})
        predictions.append(predicted); gains.append(b); offsets.append(a)
        flags.append(low_info); outside.append(extrapolation)
    return {"prediction": np.stack(predictions), "gain": np.stack(gains), "offset": np.stack(offsets),
            "low_information": np.stack(flags), "extrapolation": np.stack(outside), "metrics": pd.DataFrame(records)}


def spectral_checks(grid, means, cfg):
    lag_rows, peak_rows = [], []
    for j, label in enumerate(CLASSES):
        r, h = means["Renishaw"][j], means["Horiba"][j]
        # Common interior support for every lag avoids rewarding different cropping.
        margin = cfg.max_lag_cm
        center = np.arange(margin, len(grid) - margin)
        lags = sorted(range(-margin, margin + 1), key=lambda k: (abs(k), k))
        scores = [correlation_matrix(r[center][None, :], h[center + lag][None, :])[0, 0] for lag in lags]
        best = int(np.nanargmax(scores))
        lag_rows.append({"class": label, "best_horiba_peak_lag_cm": lags[best],
                         "correlation_same_support_lag0": scores[0], "best_correlation": scores[best],
                         "search_boundary": abs(lags[best]) == margin})
        for instrument in INSTRUMENTS:
            spectrum = means[instrument][j]
            smooth = savgol_filter(spectrum, 11, 3)
            peaks, props = find_peaks(smooth, prominence=0.05 * np.ptp(smooth), distance=10)
            if not len(peaks):
                continue
            selected = np.argsort(-props["prominences"], kind="stable")[:6]
            widths = peak_widths(smooth, peaks, rel_height=0.5)[0]
            for p in selected:
                peak_rows.append({"class": label, "instrument": instrument, "peak_cm": grid[peaks[p]],
                                  "height": float(smooth[peaks[p]]), "prominence": props["prominences"][p],
                                  "width_half_prominence_cm": widths[p]})
    return pd.DataFrame(lag_rows), pd.DataFrame(peak_rows)


def save_figure(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=140, facecolor="white", metadata={"Software": "raman spectral diagnostics"})
    plt.close(fig)


def plot_atlas(grid, means, sds, counts, view, out):
    fig, axes = plt.subplots(len(CLASSES), 1, figsize=(12, 14), sharex=True, sharey=True, layout="constrained")
    for j, (label, ax) in enumerate(zip(CLASSES, axes)):
        for instrument, color in zip(INSTRUMENTS, COLORS):
            mu, sd = means[instrument][j], sds[instrument][j]
            ax.fill_between(grid, mu - sd, mu + sd, color=color, alpha=0.13, linewidth=0)
            ax.plot(grid, mu, color=color, lw=1, label=instrument)
        sizes = counts.loc[counts["class"] == label, "selected"].tolist()
        ax.set_title(f"{label}  ·  n={sizes[0]} Renishaw / {sizes[1]} Horiba", loc="left", fontsize=10)
        ax.axhline(0, color="0.7", lw=0.4)
    axes[0].legend(frameon=False, ncol=2, loc="upper right")
    axes[-1].set(xlabel="Raman shift (cm⁻¹)", xlim=(grid[0], grid[-1]))
    fig.supylabel("Intensity (as-loaded units)" if view != "asls_minmax" else "Per-spectrum min–max intensity")
    fig.suptitle(f"{view}: molecule means ± 1 spectrum SD\nShared axes; bands describe observed spread, not confidence intervals", fontsize=13)
    save_figure(fig, out, f"{view}_spectral_atlas")


def plot_distances(means, view, out):
    pairs = (("Renishaw", "Renishaw"), ("Horiba", "Horiba"), ("Renishaw", "Horiba"))
    rmses = [rmse_matrix(means[a], means[b]) for a, b in pairs]
    correlations = [correlation_matrix(means[a], means[b]) for a, b in pairs]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), layout="constrained")
    rows = []
    for k, ((a, b), rm, co) in enumerate(zip(pairs, rmses, correlations)):
        for row, values, title, cmap, low, high in ((0, rm, "RMSE · lower is closer", "Blues", 0, max(m.max() for m in rmses)),
                                                  (1, co, "Pearson r · higher is closer", "RdBu_r", -1, 1)):
            ax = axes[row, k]
            im = ax.imshow(values, cmap=cmap, vmin=low, vmax=high, interpolation="nearest")
            ax.set(xticks=range(7), yticks=range(7), xticklabels=CLASSES, yticklabels=CLASSES,
                   xlabel=b, ylabel=a, title=f"{a} × {b}\n{title}")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            for i in range(7):
                for j in range(7):
                    t = (values[i, j] - low) / max(high - low, 1e-12)
                    color = "white" if (t > .65 if row == 0 else t < .2 or t > .8) else "black"
                    ax.text(j, i, f"{values[i,j]:.2f}" if row else f"{values[i,j]:.2g}", ha="center", va="center", fontsize=7, color=color)
            if k == 2:
                fig.colorbar(im, ax=axes[row, :], shrink=.75)
        for i, label_a in enumerate(CLASSES):
            for j, label_b in enumerate(CLASSES):
                rows.append({"instrument_a": a, "class_a": label_a, "instrument_b": b, "class_b": label_b,
                             "rmse": rm[i, j], "pearson_r": co[i, j]})
    fig.suptitle(f"{view}: class-mean comparisons, not classification accuracies\nColor scales shared across instruments within each metric", fontsize=13)
    save_figure(fig, out, f"{view}_distance_matrices")
    return pd.DataFrame(rows)


def plot_response(grid, means, sds, result, cfg, view, out):
    prediction = result["prediction"]
    target, source = means["Horiba"], means["Renishaw"]
    fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True, sharey=True, layout="constrained")
    for j, ax in enumerate(axes):
        ax.fill_between(grid, target[j] - sds["Horiba"][j], target[j] + sds["Horiba"][j], color=COLORS[1], alpha=.13, linewidth=0)
        ax.plot(grid, target[j], color=COLORS[1], lw=1, label="Observed Horiba")
        ax.plot(grid, prediction[j], color=COLORS[0], lw=1, label="Prediction; molecule excluded from fit")
        ax.plot(grid, source[j], color="0.55", lw=.65, alpha=.7, label="Uncorrected Renishaw")
        ax.set_title(CLASSES[j], loc="left", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8, ncol=3, loc="upper right")
    axes[-1].set(xlabel="Raman shift (cm⁻¹)", xlim=(grid[0], grid[-1]))
    fig.supylabel("Intensity (min–max units)" if view == "asls_minmax" else "Intensity (as-loaded units)")
    fig.suptitle(f"{view}: leave-one-molecule-out shared response, Renishaw → Horiba\nFixed {cfg.coefficient_window}-point smoothing; bands = target spectrum SD", fontsize=13)
    save_figure(fig, out, f"{view}_held_out_predictions")

    residual = prediction - target
    standardized = np.divide(residual, sds["Horiba"], out=np.full_like(residual, np.nan), where=sds["Horiba"] > 0)
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, layout="constrained")
    im = axes[0].imshow(standardized, aspect="auto", extent=(grid[0]-.5, grid[-1]+.5, 6.5, -.5), cmap="RdBu_r", vmin=-3, vmax=3, interpolation="nearest")
    axes[0].set_title("Held-out residual / target spectrum SD · color clipped at ±3; full values saved", loc="left")
    fig.colorbar(im, ax=axes[0], label="Prediction − observation, in target SD", extend="both")
    for ax, values, title in ((axes[1], result["low_information"], f"Weak training contrast: between-molecule spread ≤ {cfg.low_information_ratio:g} × within-molecule spread (dark)"),
                              (axes[2], result["extrapolation"], "Held-out source intensity outside training-molecule range (dark)")):
        ax.imshow(values.astype(float), aspect="auto", extent=(grid[0]-.5, grid[-1]+.5, 6.5, -.5), cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, loc="left", fontsize=10)
    for ax in axes:
        ax.set(yticks=range(7), yticklabels=CLASSES)
    axes[-1].set_xlabel("Raman shift (cm⁻¹)")
    fig.suptitle(f"{view}: residual and reference-information maps\nDescriptive flags, not hypothesis tests or proof of identifiability", fontsize=13)
    save_figure(fig, out, f"{view}_residual_map")

    a, b, _ = fit_shared_response(source, target, sds["Renishaw"], cfg)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for ax, w in zip(axes.flat, cfg.scatter_wavenumbers):
        k = int(np.argmin(abs(grid - w)))
        xr = source[:, k]
        ax.errorbar(xr, target[:, k], xerr=sds["Renishaw"][:, k], yerr=sds["Horiba"][:, k],
                    fmt="o", markersize=4, color=COLORS[0], ecolor="0.82", elinewidth=.7)
        line_x = np.linspace(xr.min(), xr.max(), 100)
        ax.plot(line_x, a[k] + b[k] * line_x, color="0.35", lw=1)
        for j, label in enumerate(CLASSES):
            ax.annotate(label, (xr[j], target[j, k]), xytext=(4, 4), textcoords="offset points", fontsize=7)
        ax.set(title=f"{grid[k]:g} cm⁻¹", xlabel="Renishaw mean intensity", ylabel="Horiba mean intensity")
    fig.suptitle(f"{view}: shared response at six prespecified coordinates\nAll-molecule descriptive fit; error bars = spectrum SD, not uncertainty of the mean", fontsize=13)
    save_figure(fig, out, f"{view}_response_scatter")
    return standardized


def run_diagnostics(data_dir, output_root, cfg=None):
    cfg = cfg or DiagnosticConfig()
    if cfg.seed < 0 or cfg.max_per_class < 2 or cfg.high - cfg.low < cfg.coefficient_window:
        raise ValueError("Invalid seed, sample size, or spectral interval")
    if cfg.coefficient_window % 2 != 1 or cfg.coefficient_polyorder >= cfg.coefficient_window:
        raise ValueError("Coefficient smoothing requires an odd window larger than polynomial order")
    data_dir, output_root = Path(data_dir), Path(output_root)
    with threadpool_limits(limits=1), plt.rc_context({"axes.spines.top": False, "axes.spines.right": False,
                                                    "axes.grid": False, "font.family": "DejaVu Sans", "font.size": 10}):
        grid, raw, selection, counts, inputs = load_inputs(data_dir, cfg)
        environment = {"python": sys.version, "platform": platform.platform(),
                       "packages": {p: version(p) for p in ("numpy", "pandas", "scipy", "matplotlib", "pybaselines", "threadpoolctl")}}
        identity = {"config": asdict(cfg), "inputs": inputs, "environment": environment,
                    "analysis_code_sha256": file_hash(__file__)}
        run_id = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        out = output_root / run_id
        out.mkdir(parents=True, exist_ok=True)
        selection.to_csv(out / "selected_rows.csv", index=False)
        counts.to_csv(out / "input_counts.csv", index=False)
        print(f"Diagnostic run {run_id}: {int(counts.selected.sum()):,} selected spectra", flush=True)
        views, preprocessing_audit = preprocess(raw, cfg)
        preprocessing_audit.to_csv(out / "preprocessing_audit.csv", index=False)
        arrays, metric_tables = {"wavenumbers": grid}, []
        for view in VIEWS:
            print(f"Analyzing {view}", flush=True)
            means, sds = moments(views[view])
            result = leave_one_molecule_out(means, sds, cfg)
            metrics = result["metrics"].assign(view=view)
            metric_tables.append(metrics)
            plot_atlas(grid, means, sds, counts, view, out)
            distances = plot_distances(means, view, out)
            standardized = plot_response(grid, means, sds, result, cfg, view, out)
            lag, peaks = spectral_checks(grid, means, cfg)
            for name, table in (("distances", distances), ("held_out_metrics", metrics), ("lag_screen", lag), ("peak_screen", peaks)):
                table.to_csv(out / f"{view}_{name}.csv", index=False, float_format="%.17g")
            for instrument in INSTRUMENTS:
                arrays[f"{view}_{instrument}_mean"] = means[instrument]
                arrays[f"{view}_{instrument}_sd"] = sds[instrument]
            for key in ("prediction", "gain", "offset", "low_information", "extrapolation"):
                arrays[f"{view}_{key}"] = result[key]
            arrays[f"{view}_standardized_residual"] = standardized
        metrics = pd.concat(metric_tables, ignore_index=True)
        metrics.to_csv(out / "held_out_metrics_all_views.csv", index=False, float_format="%.17g")
        np.savez_compressed(out / "numerical_results.npz", **arrays)
        fingerprints = {key: array_hash(value) for key, value in arrays.items()}
        manifest = {**identity, "run_id": run_id, "class_order": CLASSES, "instrument_order": INSTRUMENTS,
                    "selection_sha256": file_hash(out / "selected_rows.csv"), "array_sha256": fingerprints,
                    "selected_input_sha256": {f"{i}/{c}": array_hash(raw[i, c]) for i in INSTRUMENTS for c in CLASSES},
                    "notes": ["As-loaded CSV lipid/RNA provenance may include earlier processing; not claimed to be raw detector data.",
                              "RNA/Lipid labels inherit notebook row-order convention: first 375 RNA, next 375 Lipid.",
                              "No quality filtering, carbon-region deletion, KNN pairing, GPU, model training, or bootstrap.",
                              "Independent local PCG64 streams select rows; single-thread numerical libraries; float64 arithmetic.",
                              "SD is within selected spectra; correlated acquisition rows are not independent biological replicates.",
                              "Held-out fits exclude that molecule from both instruments; held-out Horiba is evaluation only.",
                              "Shape/scatter/lag/peak diagnostics are descriptive; no automatic registration or resolution correction.",
                              "Negative gains are allowed and reported; this is a diagnostic hypothesis, not a validated physical calibration.",
                              "Identical numerical results are expected for identical inputs, configuration, code and environment; cross-platform bitwise identity is not guaranteed."]}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (out / "environment.txt").write_text("\n".join(f"{p}=={v}" for p, v in environment["packages"].items()) + "\n")
        (out / "README.md").write_text("# Biomolecule spectral diagnostics\n\n"
            "Start with the asls_minmax spectral atlas, distance matrices, held-out predictions, and residual map; compare with asls and as_loaded to inspect preprocessing effects.\n\n"
            "Each fold learns the shared Renishaw-to-Horiba affine response from six molecule means and evaluates the seventh. Coefficients use fixed 31-point smoothing by default. Negative RMSE change is an improvement in spectral mean prediction, not classification.\n\n"
            "The lag screen and half-prominence peak widths are exploratory checks for alignment/resolution mismatch, not an automatic correction. Positive lag means a Horiba feature lies at a higher wavenumber.\n\n"
            "Residual/SD maps compare mean prediction error with observed target-spectrum spread, not a confidence interval or significance threshold. Weak-contrast and extrapolation flags are diagnostic warnings, not impossibility certificates.\n\n"
            "See manifest.json for configuration, input hashes, numerical fingerprints, provenance caveats, and environment. selected_rows.csv records exact selected source-row identities.\n")
        return {"output_dir": out, "run_id": run_id, "counts": counts, "metrics": metrics,
                "preprocessing_audit": preprocessing_audit, "fingerprints": fingerprints, "manifest": manifest}


def verify_reproducibility(first, second):
    if first["run_id"] != second["run_id"]:
        raise AssertionError("Run identities differ: input/config/code/environment changed")
    if first["manifest"]["selection_sha256"] != second["manifest"]["selection_sha256"]:
        raise AssertionError("Selected rows changed")
    if first["manifest"]["selected_input_sha256"] != second["manifest"]["selected_input_sha256"]:
        raise AssertionError("Selected input arrays changed")
    if first["fingerprints"] != second["fingerprints"]:
        raise AssertionError("Numerical outputs changed")
    pd.testing.assert_frame_equal(first["metrics"], second["metrics"], check_exact=True)
    return "PASS: identical selected rows, selected input arrays, all numerical output arrays, and metric tables."
