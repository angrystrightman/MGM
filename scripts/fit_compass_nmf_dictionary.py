#!/usr/bin/env python3
"""Fit an NMF taxa dictionary for ComPASS-Biome warm starts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from scripts.compass_biome_utils import reconstruction_metric_rows, top_taxa_rows


def _load_taxa_matrix(dataset_dir: Path) -> np.ndarray:
    path = dataset_dir / "X_taxa.npz"
    if not path.exists():
        path = dataset_dir / "X_unnorm.npz"
    with np.load(path, allow_pickle=False) as payload:
        return payload["X"].astype(np.float32)


def _load_splits(dataset_dir: Path, sample_ids: Sequence[str]) -> list[str]:
    split_path = dataset_dir / "sample_splits.csv"
    if not split_path.exists():
        return ["train"] * len(sample_ids)
    splits = pd.read_csv(split_path)
    if "sample_id" not in splits.columns or "split" not in splits.columns:
        raise ValueError("sample_splits.csv must contain sample_id and split columns")
    split_ids = splits["sample_id"].astype(str).tolist()
    if split_ids != list(sample_ids):
        raise ValueError("sample_splits.csv sample IDs do not match dataset sample IDs")
    return splits["split"].astype(str).tolist()


def _normalize_rows(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=np.float32),
        where=row_sums > eps,
    ).astype(np.float32)


def fit_nmf_dictionary(
    dataset_dir: Path,
    output_dir: Path,
    num_programs: int = 32,
    max_iter: int = 500,
    seed: int = 0,
    top_k: int = 20,
) -> dict[str, object]:
    """Fit train-split-only NMF and write dictionary/oracle reports."""
    from sklearn.decomposition import NMF

    if num_programs <= 0:
        raise ValueError("num_programs must be positive")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")

    sample_ids = pd.read_csv(dataset_dir / "sample_ids.csv")["sample_id"].astype(str).tolist()
    taxa_names = pd.read_csv(dataset_dir / "taxa_names.csv")["taxon"].astype(str).tolist()
    x = _load_taxa_matrix(dataset_dir)
    splits = _load_splits(dataset_dir, sample_ids)
    split_array = np.asarray(splits, dtype=str)
    train_mask = split_array == "train"
    if not np.any(train_mask):
        raise ValueError("NMF fitting requires at least one train sample")
    if x.ndim != 2:
        raise ValueError("taxa matrix must be 2D")
    if x.shape[0] != len(sample_ids):
        raise ValueError("taxa matrix row count must match sample IDs")
    if x.shape[1] != len(taxa_names):
        raise ValueError("taxa matrix width must match taxa names")
    if np.any(x < 0) or not np.isfinite(x).all():
        raise ValueError("taxa matrix must be finite and non-negative")

    model = NMF(
        n_components=num_programs,
        init="nndsvda",
        solver="mu",
        beta_loss="kullback-leibler",
        max_iter=max_iter,
        random_state=seed,
    )
    x_train = x[train_mask].astype(np.float32, copy=False)
    model.fit_transform(x_train)
    components_raw = model.components_.astype(np.float32)
    taxa_programs = _normalize_rows(components_raw)

    sample_weights = model.transform(x.astype(np.float32, copy=False)).astype(np.float32)
    oracle_reconstruction = (sample_weights @ components_raw).astype(np.float32)
    oracle_reconstruction = _normalize_rows(oracle_reconstruction)
    metrics = pd.DataFrame(
        reconstruction_metric_rows(
            x,
            oracle_reconstruction,
            splits,
            model_name="nmf_oracle",
            top_k=top_k,
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "nmf_dictionary.npz",
        taxa_programs=taxa_programs,
        components_raw=components_raw,
        taxa_names=np.asarray(taxa_names),
    )
    metrics.to_csv(output_dir / "nmf_oracle_metrics.csv", index=False)
    pd.DataFrame(top_taxa_rows(taxa_programs, taxa_names, top_k=top_k)).to_csv(
        output_dir / "top_taxa_per_program.csv",
        index=False,
    )

    manifest = {
        "dataset_dir": str(dataset_dir),
        "num_samples": int(x.shape[0]),
        "num_train_samples": int(train_mask.sum()),
        "num_taxa": int(x.shape[1]),
        "num_programs": int(num_programs),
        "max_iter": int(max_iter),
        "seed": int(seed),
        "init": "nndsvda",
        "solver": "mu",
        "beta_loss": "kullback-leibler",
        "n_iter": int(model.n_iter_),
        "reconstruction_err": float(model.reconstruction_err_),
        "split_counts": pd.Series(splits).value_counts().sort_index().to_dict(),
        "outputs": {
            "nmf_dictionary": str(output_dir / "nmf_dictionary.npz"),
            "nmf_oracle_metrics": str(output_dir / "nmf_oracle_metrics.csv"),
            "top_taxa_per_program": str(output_dir / "top_taxa_per_program.csv"),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-programs", type=int, default=32)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = fit_nmf_dictionary(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        num_programs=args.num_programs,
        max_iter=args.max_iter,
        seed=args.seed,
        top_k=args.top_k,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
