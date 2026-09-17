"""Tune the notebook's patch Transformer on seven biomolecule Raman classes.

Run from the project folder with: .venv/bin/python optuna_biomolecules.py
The data source is RAMAN_DATA_DIR in .env, exactly as in the notebook.
"""

import json
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import pybaselines
import torch
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
DATA = Path(os.environ.get("RAMAN_DATA_DIR", ROOT / "data")).expanduser()
OUT = ROOT / "optuna_results"
CLASSES = ["adenine", "dl-ala", "dl-phe", "dl-tyr", "glu", "RNA", "Lipid"]
MOLECULES = CLASSES[:5]
GRID = np.arange(450, 1451)
KEEP = (GRID < 1250) | (GRID > 1450)
SEED = 42


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def finish_spectra(values):
    """Reproduce notebook ASLS, row-wise min-max, and carbon-peak removal."""
    corrected = np.stack([
        row - pybaselines.whittaker.asls(row, lam=1e5, p=0.01)[0]
        for row in values.astype(float)
    ])
    lo = corrected.min(axis=1, keepdims=True)
    span = corrected.max(axis=1, keepdims=True) - lo
    normalized = (corrected - lo) / np.where(span == 0, 1, span)
    return normalized[:, KEEP].astype(np.float32)


def read_molecule(instrument, label):
    path = DATA / f"{label}_{instrument}.txt"
    if instrument == "horiba":
        frame = pd.read_csv(path, sep="\t").iloc[:, 2:]
        wn = frame.columns.astype(float).to_numpy()
        raw = frame.to_numpy(dtype=float)
    else:
        frame = pd.read_csv(path, sep="\t", header=None).iloc[:, 2:]
        frame.columns = ["wavenumber", "intensity"]
        frame["index"] = frame.groupby("wavenumber").cumcount()
        pivot = frame.pivot(index="index", columns="wavenumber", values="intensity").sort_index(axis=1)
        wn = pivot.columns.astype(float).to_numpy()
        raw = pivot.to_numpy(dtype=float)
    if len(raw) > 750:
        raw = raw[np.random.default_rng(SEED).choice(len(raw), 750, replace=False)]
    interpolated = np.stack([np.interp(GRID, wn, row) for row in raw])
    return finish_spectra(interpolated)


def read_lipid_rna(instrument):
    frame = pd.read_csv(DATA / f"{instrument}_lipid_rna.csv")
    # Notebook assigns rows 0:374 to RNA, remaining rows to Lipid.
    spectral = frame.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    names = np.where(np.arange(len(frame)) < 375, "RNA", "Lipid")
    # Original notebook keeps these columns in source-file order.
    columns = np.asarray(frame.columns, dtype=float)
    if not np.array_equal(columns, GRID):
        raise ValueError(f"Unexpected lipid/RNA spectral grid in {instrument}: {columns[:3]}")
    return finish_spectra(spectral), names


def load_data(instrument):
    chunks, labels = [], []
    for label in MOLECULES:
        x = read_molecule(instrument, label)
        chunks.append(x)
        labels.extend([CLASSES.index(label)] * len(x))
    x, names = read_lipid_rna(instrument)
    chunks.append(x)
    labels.extend([CLASSES.index(name) for name in names])
    return np.concatenate(chunks), np.asarray(labels, dtype=np.int64)


class RamanTransformer(nn.Module):
    """Same patch-embedding/CLS/encoder structure as notebook cells 58/103."""

    def __init__(self, length, patch_size, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.patch_size = patch_size
        patches = (length + patch_size - 1) // patch_size
        self.patch_embed = nn.Conv1d(1, d_model, patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.zeros(1, patches + 1, d_model))
        self.embedding_dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, len(CLASSES)))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x):
        x = x.unsqueeze(1)
        pad = (-x.shape[-1]) % self.patch_size
        if pad:
            x = F.pad(x, (0, pad), mode="replicate")
        tokens = self.patch_embed(x).transpose(1, 2)
        cls = self.cls_token.expand(len(x), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embedding[:, :tokens.shape[1]]
        encoded = self.encoder(self.embedding_dropout(tokens))[:, 0]
        return self.classifier(encoded)


def predict(model, x, device, batch_size=256):
    model.eval()
    results = []
    with torch.no_grad():
        for (batch,) in DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size):
            results.append(model(batch.to(device)).argmax(1).cpu().numpy())
    return np.concatenate(results)


def train_model(params, x_train, y_train, x_val, y_val, device, max_epochs, trial=None, refit=False):
    seed_everything(SEED)
    model = RamanTransformer(x_train.shape[1], **{k: params[k] for k in
        ["patch_size", "d_model", "nhead", "num_layers", "dim_feedforward", "dropout"]}).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                        batch_size=params["batch_size"], shuffle=True)
    best_acc, best_epoch, best_state = -1, 0, None
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_acc = accuracy_score(y_val, predict(model, x_val, device)) if not refit else float("nan")
        if refit or val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"epoch {epoch+1}/{max_epochs}: validation accuracy {val_acc:.4f}", flush=True)
        if trial is not None:
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if not refit and epoch - best_epoch >= 3:
            break
    model.load_state_dict(best_state)
    return model, best_acc, best_epoch


def save_matrix(y_true, y_pred, path, title):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASSES)))
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(CLASSES)), yticks=np.arange(len(CLASSES)),
           xticklabels=CLASSES, yticklabels=CLASSES, xlabel="Predicted", ylabel="True", title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return cm.tolist()


def run_study(source_x, source_y, target_x, target_y, trials=8, epochs=8):
    """Tune/evaluate already-prepared frames; callable from a VS Code notebook cell."""
    OUT.mkdir(exist_ok=True)
    seed_everything(SEED)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    source_x, source_y = load_data("renishaw")
    target_x, target_y = load_data("horiba")
    indices = np.arange(len(source_y))
    train_idx, hold_idx = train_test_split(indices, test_size=0.4, random_state=SEED, stratify=source_y)
    val_idx, test_idx = train_test_split(hold_idx, test_size=0.5, random_state=SEED, stratify=source_y[hold_idx])
    print("Device:", device, "source:", source_x.shape, "target:", target_x.shape, flush=True)
    print("Class counts source:", np.bincount(source_y), "target:", np.bincount(target_y), flush=True)

    def objective(trial):
        d_model = trial.suggest_categorical("d_model", [32, 64])
        params = {
            "patch_size": trial.suggest_categorical("patch_size", [32, 64]),
            "d_model": d_model,
            "nhead": trial.suggest_categorical("nhead", [4, 8]),
            "num_layers": trial.suggest_int("num_layers", 1, 2),
            "dim_feedforward": trial.suggest_categorical("dim_feedforward", [64, 128]),
            "dropout": trial.suggest_float("dropout", 0.05, 0.3),
            "lr": trial.suggest_float("lr", 1e-4, 2e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128]),
        }
        _, acc, best_epoch = train_model(params, source_x[train_idx], source_y[train_idx],
                                         source_x[val_idx], source_y[val_idx], device, epochs, trial)
        trial.set_user_attr("best_epoch", best_epoch)
        return acc

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=3))
    study.optimize(objective, n_trials=trials)
    study.trials_dataframe().to_csv(OUT / "trials.csv", index=False)
    best = study.best_trial
    # Refit on train + validation, for the epoch count picked on validation only.
    fit_idx = np.concatenate([train_idx, val_idx])
    model, _, _ = train_model(best.params, source_x[fit_idx], source_y[fit_idx],
                              source_x[val_idx], source_y[val_idx], device,
                              best.user_attrs["best_epoch"], refit=True)
    source_pred = predict(model, source_x[test_idx], device)
    target_pred = predict(model, target_x, device)
    result = {
        "classes": CLASSES,
        "source_counts": np.bincount(source_y).tolist(),
        "target_counts": np.bincount(target_y).tolist(),
        "split_counts": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
        "best_trial": best.number, "best_parameters": best.params,
        "validation_accuracy": best.value, "best_epoch": best.user_attrs["best_epoch"],
        "renishaw_test_accuracy": accuracy_score(source_y[test_idx], source_pred),
        "horiba_accuracy": accuracy_score(target_y, target_pred),
        "renishaw_confusion_matrix": save_matrix(source_y[test_idx], source_pred, OUT / "renishaw_confusion.png", "Renishaw held-out test"),
        "horiba_confusion_matrix": save_matrix(target_y, target_pred, OUT / "horiba_confusion.png", "Horiba external instrument"),
    }
    torch.save(model.cpu().state_dict(), OUT / "best_transformer.pt")
    (OUT / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result


def run_from_notebook(renishaw_everything, horiba_everything, trials=8, epochs=8):
    """Use the two frames after the notebook's carbon-peak-cut cell (no reprocessing)."""
    metadata = {"Virus", "Subtype", "Strain"}
    spectral_columns = sorted(
        set(renishaw_everything.columns).intersection(horiba_everything.columns) - metadata,
        key=float,
    )
    source = renishaw_everything.loc[renishaw_everything["Virus"].isin(CLASSES)]
    target = horiba_everything.loc[horiba_everything["Virus"].isin(CLASSES)]
    if len(spectral_columns) != int(KEEP.sum()):
        raise ValueError(f"Expected {KEEP.sum()} post-cut spectral columns; got {len(spectral_columns)}")
    source_x = source[spectral_columns].to_numpy(dtype=np.float32)
    source_y = source["Virus"].map(CLASSES.index).to_numpy(dtype=np.int64)
    target_x = target[spectral_columns].to_numpy(dtype=np.float32)
    target_y = target["Virus"].map(CLASSES.index).to_numpy(dtype=np.int64)
    return run_study(source_x, source_y, target_x, target_y, trials=trials, epochs=epochs)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    run_study(*load_data("renishaw"), *load_data("horiba"), trials=args.trials, epochs=args.epochs)


if __name__ == "__main__":
    main()
