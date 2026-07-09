#!/usr/bin/env python3
"""Shared helpers for ComPASS-Biome MicroCorpus-260K pilot scripts."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


BIOME_COLUMNS = ["biome_1", "biome_2", "biome_3", "biome_4", "biome_5"]


def normalize_mgnify_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    """Rename MGM's compact MGnify biome CSV columns to stable names."""
    rename_map = {"Unnamed: 0": "sample_id", "0": "sample_id"}
    for idx, name in enumerate(BIOME_COLUMNS, start=1):
        rename_map[str(idx)] = name
        rename_map[idx] = name

    normalized = metadata.rename(columns=rename_map).copy()
    if "sample_id" not in normalized.columns:
        first_col = normalized.columns[0]
        normalized = normalized.rename(columns={first_col: "sample_id"})

    expected = ["sample_id"] + [col for col in BIOME_COLUMNS if col in normalized.columns]
    remaining = [col for col in normalized.columns if col not in expected]
    normalized = normalized[expected + remaining]
    normalized["sample_id"] = normalized["sample_id"].astype(str)
    return normalized


def duplicate_values(values: Iterable[str]) -> list[str]:
    counts = Counter(values)
    return sorted([value for value, count in counts.items() if count > 1])


def alignment_report(corpus_ids: Sequence[str], metadata_ids: Sequence[str]) -> dict[str, object]:
    """Return conservative sample-ID alignment counts."""
    corpus_list = [str(value) for value in corpus_ids]
    metadata_list = [str(value) for value in metadata_ids]
    corpus_set = set(corpus_list)
    metadata_set = set(metadata_list)

    return {
        "num_corpus_samples": len(corpus_list),
        "num_metadata_rows": len(metadata_list),
        "num_matched_samples": len(corpus_set & metadata_set),
        "num_unmatched_in_corpus": len(corpus_set - metadata_set),
        "num_unmatched_in_metadata": len(metadata_set - corpus_set),
        "duplicate_corpus_sample_ids": duplicate_values(corpus_list),
        "duplicate_metadata_sample_ids": duplicate_values(metadata_list),
        "ordering_matches": corpus_list == metadata_list,
    }


def select_balanced_sample_ids(
    metadata: pd.DataFrame,
    label_column: str,
    max_samples: int,
    seed: int = 0,
) -> list[str]:
    """Select sample IDs by round-robin sampling across non-null label groups."""
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if "sample_id" not in metadata.columns:
        raise ValueError("metadata must contain a sample_id column")
    if label_column not in metadata.columns:
        raise ValueError(f"metadata must contain label column {label_column!r}")

    rng = np.random.default_rng(seed)
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in metadata[["sample_id", label_column]].dropna().itertuples(index=False):
        grouped[str(row[1])].append(str(row[0]))
    if not grouped:
        raise ValueError(f"No non-null labels found for {label_column!r}")

    for key in grouped:
        grouped[key] = list(rng.permutation(grouped[key]))

    labels = sorted(grouped)
    selected: list[str] = []
    while len(selected) < max_samples and any(grouped.values()):
        for label in labels:
            if len(selected) >= max_samples:
                break
            if grouped[label]:
                selected.append(grouped[label].pop())
    return selected


def _ratio_counts(n_items: int, ratios: Sequence[float]) -> list[int]:
    if n_items < 0:
        raise ValueError("n_items must be non-negative")
    if not ratios or any(ratio < 0 for ratio in ratios):
        raise ValueError("ratios must be non-empty and non-negative")
    total = float(sum(ratios))
    if total <= 0:
        raise ValueError("ratios must sum to a positive value")

    raw = np.array([ratio / total * n_items for ratio in ratios], dtype=np.float64)
    counts = np.floor(raw).astype(int)
    remaining = int(n_items - counts.sum())
    order = sorted(range(len(ratios)), key=lambda idx: (-(raw[idx] - counts[idx]), idx))
    for idx in order[:remaining]:
        counts[idx] += 1

    if n_items >= len(ratios):
        for idx in range(len(counts)):
            if counts[idx] == 0:
                donor = int(np.argmax(counts))
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[idx] = 1
    return counts.tolist()


def stratified_split_sample_ids(
    metadata: pd.DataFrame,
    sample_ids: Sequence[str],
    label_column: str,
    seed: int = 0,
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
) -> pd.DataFrame:
    """Create deterministic train/valid/test assignments stratified by a metadata label."""
    sample_ids = [str(sample_id) for sample_id in sample_ids]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_ids must be unique")
    if "sample_id" not in metadata.columns:
        raise ValueError("metadata must contain a sample_id column")
    if label_column not in metadata.columns:
        raise ValueError(f"metadata must contain label column {label_column!r}")
    if len(ratios) != 3:
        raise ValueError("ratios must contain train, valid, and test weights")

    metadata_by_id = metadata.copy()
    metadata_by_id["sample_id"] = metadata_by_id["sample_id"].astype(str)
    metadata_by_id = metadata_by_id.set_index("sample_id", drop=False)
    missing = [sample_id for sample_id in sample_ids if sample_id not in metadata_by_id.index]
    if missing:
        raise ValueError(f"sample IDs missing from metadata: {missing[:10]}")

    selected_metadata = metadata_by_id.loc[sample_ids, ["sample_id", label_column]].copy()
    selected_metadata[label_column] = selected_metadata[label_column].astype("string").fillna("__missing__")

    rng = np.random.default_rng(seed)
    split_by_id: dict[str, str] = {}
    for label in sorted(selected_metadata[label_column].unique()):
        group_ids = selected_metadata.loc[selected_metadata[label_column] == label, "sample_id"].tolist()
        group_ids = [str(value) for value in rng.permutation(group_ids)]
        train_count, valid_count, test_count = _ratio_counts(len(group_ids), ratios)
        split_labels = (
            ["train"] * train_count
            + ["valid"] * valid_count
            + ["test"] * test_count
        )
        for sample_id, split in zip(group_ids, split_labels):
            split_by_id[sample_id] = split

    rows = []
    for sample_id in sample_ids:
        label = selected_metadata.loc[sample_id, label_column]
        rows.append(
            {
                "sample_id": sample_id,
                label_column: str(label),
                "split": split_by_id[sample_id],
            }
        )
    return pd.DataFrame(rows)


def select_top_taxa_by_train(
    x: np.ndarray,
    splits: Sequence[str],
    taxa_names: Sequence[str],
    num_taxa: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Select taxa using train split prevalence, with mean abundance tie-breaks."""
    x = np.asarray(x, dtype=np.float32)
    splits = np.asarray([str(split) for split in splits])
    if x.ndim != 2:
        raise ValueError("x must be a 2D array")
    if x.shape[0] != len(splits):
        raise ValueError("splits length must match x rows")
    if x.shape[1] != len(taxa_names):
        raise ValueError("taxa_names length must match x columns")
    if num_taxa <= 0:
        raise ValueError("num_taxa must be positive")
    if not np.any(splits == "train"):
        raise ValueError("at least one train sample is required")

    train_x = x[splits == "train"]
    prevalence = (train_x > 0).mean(axis=0)
    mean_abundance = train_x.mean(axis=0)
    order = sorted(
        range(x.shape[1]),
        key=lambda idx: (-float(prevalence[idx]), -float(mean_abundance[idx]), str(taxa_names[idx])),
    )[: min(num_taxa, x.shape[1])]
    indices = np.array(order, dtype=np.int64)
    table = pd.DataFrame(
        {
            "rank": np.arange(1, len(indices) + 1, dtype=np.int64),
            "taxon": [str(taxa_names[idx]) for idx in indices],
            "source_index": indices,
            "train_prevalence": [float(prevalence[idx]) for idx in indices],
            "train_mean_abundance": [float(mean_abundance[idx]) for idx in indices],
        }
    )
    return indices, table


def filter_and_normalize_taxa(x: np.ndarray, selected_indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Filter taxa columns and renormalize rows within the selected taxa universe."""
    x = np.asarray(x, dtype=np.float32)
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    if x.ndim != 2:
        raise ValueError("x must be a 2D array")
    if selected_indices.ndim != 1:
        raise ValueError("selected_indices must be 1D")
    if len(selected_indices) == 0:
        raise ValueError("selected_indices must not be empty")

    selected = x[:, selected_indices].astype(np.float32, copy=True)
    retained_mass = selected.sum(axis=1).astype(np.float32)
    normalized = np.divide(
        selected,
        retained_mass[:, None],
        out=np.zeros_like(selected, dtype=np.float32),
        where=retained_mass[:, None] > 0,
    )
    return normalized.astype(np.float32), retained_mass


def task_suitability(
    metadata: pd.DataFrame,
    has_corpus: bool,
    has_unnorm_data: bool,
) -> dict[str, dict[str, str]]:
    """Classify what the inspected data can support."""
    lower_columns = {str(col).lower() for col in metadata.columns}
    has_biome = any(col in metadata.columns for col in BIOME_COLUMNS)
    has_study = any("study" in col or "project" in col or "cohort" in col for col in lower_columns)
    has_country = any("country" in col or "region" in col or "location" in col for col in lower_columns)
    has_platform = any("platform" in col or "instrument" in col for col in lower_columns)
    has_disease = any(
        "disease" in col or "phenotype" in col or "diagnosis" in col or "case" in col
        for col in lower_columns
    )
    has_pathway = any("pathway" in col or "function" in col or "ko" == col for col in lower_columns)

    return {
        "MGM teacher embedding extraction": {
            "support": "yes" if has_corpus else "no",
            "reason": "MicroCorpus pickle is loadable." if has_corpus else "No loadable corpus.",
        },
        "taxa reconstruction target": {
            "support": "yes" if has_unnorm_data else "no",
            "reason": "Unnormalized corpus .data is sample x genus relative abundance."
            if has_unnorm_data
            else "No unnormalized abundance matrix.",
        },
        "biome/source classification": {
            "support": "yes" if has_biome else "no",
            "reason": "MGnify biome hierarchy columns are available." if has_biome else "No biome labels.",
        },
        "body-site classification": {
            "support": "partial" if has_biome else "no",
            "reason": "Body-site-like labels appear only inside biome hierarchy for subsets.",
        },
        "disease prediction": {
            "support": "yes" if has_disease else "no",
            "reason": "Disease/phenotype-like columns found." if has_disease else "No disease/phenotype columns.",
        },
        "leave-one-study-out validation": {
            "support": "yes" if has_study else "no",
            "reason": "Study/cohort columns found." if has_study else "No explicit study/cohort column.",
        },
        "country/geography confounder control": {
            "support": "yes" if has_country else "no",
            "reason": "Geographic columns found." if has_country else "No country/region/location column.",
        },
        "sequencing-platform confounder control": {
            "support": "yes" if has_platform else "no",
            "reason": "Platform/instrument columns found." if has_platform else "No platform/instrument column.",
        },
        "pathway reconstruction": {
            "support": "yes" if has_pathway else "no",
            "reason": "Pathway-like columns found." if has_pathway else "No pathway abundance matrix.",
        },
        "cvaNMF/NMF baseline": {
            "support": "yes" if has_unnorm_data else "no",
            "reason": "Fixed sample x genus matrix is available.",
        },
        "program-cohort stability": {
            "support": "yes" if has_study else "no",
            "reason": "Use study/cohort labels." if has_study else "No cohort labels; use biome stability instead.",
        },
    }


def top_taxa_rows(program_taxa: np.ndarray, taxa_names: Sequence[str], top_k: int) -> list[dict[str, object]]:
    """Return top taxa rows for each program."""
    if program_taxa.ndim != 2:
        raise ValueError("program_taxa must be a 2D array")
    if program_taxa.shape[1] != len(taxa_names):
        raise ValueError("taxa_names length must match program_taxa width")

    rows: list[dict[str, object]] = []
    for program_idx, weights in enumerate(program_taxa):
        order = sorted(range(len(weights)), key=lambda idx: (-float(weights[idx]), idx))[:top_k]
        for rank, taxon_idx in enumerate(order, start=1):
            rows.append(
                {
                    "program": int(program_idx),
                    "rank": int(rank),
                    "taxon": str(taxa_names[taxon_idx]),
                    "weight": round(float(weights[taxon_idx]), 6),
                }
            )
    return rows


def top_k_recall(y_true: np.ndarray, y_pred: np.ndarray, k: int = 20) -> float:
    """Mean overlap between each row's true and predicted top-k taxa."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if y_true.ndim != 2:
        raise ValueError("inputs must be 2D arrays")
    if k <= 0:
        raise ValueError("k must be positive")

    width = y_true.shape[1]
    k_eff = min(k, width)
    scores: list[float] = []
    for true_row, pred_row in zip(y_true, y_pred):
        nonzero_true = int(np.count_nonzero(true_row > 0))
        denom = min(k_eff, nonzero_true)
        if denom == 0:
            continue
        true_top = set(np.argsort(-true_row, kind="mergesort")[:denom].tolist())
        pred_top = set(np.argsort(-pred_row, kind="mergesort")[:k_eff].tolist())
        scores.append(len(true_top & pred_top) / denom)
    return float(np.mean(scores)) if scores else float("nan")


def bray_curtis_similarity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Bray-Curtis similarity, where 1 is identical and 0 is maximally different."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if y_true.ndim != 2:
        raise ValueError("inputs must be 2D arrays")

    numerator = np.abs(y_true - y_pred).sum(axis=1)
    denominator = (np.abs(y_true) + np.abs(y_pred)).sum(axis=1)
    scores = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 0,
    )
    return float(np.mean(1.0 - scores))


def reconstruction_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(-(y_true * np.log(np.clip(y_pred, 1e-8, None))).sum(axis=1).mean())


def reconstruction_metric_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    splits: Sequence[str],
    model_name: str,
    top_k: int = 20,
) -> list[dict[str, object]]:
    """Compute reconstruction metrics by split for one model/control."""
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    splits_array = np.asarray([str(split) for split in splits])
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if y_true.shape[0] != len(splits_array):
        raise ValueError("splits length must match matrix rows")

    rows: list[dict[str, object]] = []
    for split in ["train", "valid", "test"]:
        mask = splits_array == split
        if not np.any(mask):
            continue
        true_split = y_true[mask]
        pred_split = y_pred[mask]
        rows.append(
            {
                "model": str(model_name),
                "split": split,
                "num_samples": int(mask.sum()),
                "cross_entropy": round(reconstruction_cross_entropy(true_split, pred_split), 6),
                f"top_{top_k}_recall": round(top_k_recall(true_split, pred_split, top_k), 6),
                "bray_curtis_similarity": round(bray_curtis_similarity(true_split, pred_split), 6),
            }
        )
    return rows


def mean_baseline_predictions(x: np.ndarray, splits: Sequence[str]) -> np.ndarray:
    """Predict every sample with the train-set mean abundance distribution."""
    x = np.asarray(x, dtype=np.float32)
    splits_array = np.asarray([str(split) for split in splits])
    if x.ndim != 2:
        raise ValueError("x must be a 2D array")
    if x.shape[0] != len(splits_array):
        raise ValueError("splits length must match x rows")
    train = x[splits_array == "train"]
    if len(train) == 0:
        raise ValueError("at least one train sample is required")
    train_mean = train.mean(axis=0)
    train_mean = train_mean / max(float(train_mean.sum()), 1e-12)
    return np.tile(train_mean.astype(np.float32), (x.shape[0], 1))


def program_diagnostics_rows(
    activations: np.ndarray,
    taxa_programs: np.ndarray,
    dead_threshold: float = 0.01,
) -> list[dict[str, object]]:
    """Return collapse and redundancy diagnostics for program activations and taxa dictionaries."""
    activations = np.asarray(activations, dtype=np.float64)
    taxa_programs = np.asarray(taxa_programs, dtype=np.float64)
    if activations.ndim != 2:
        raise ValueError("activations must be a 2D array")
    if taxa_programs.ndim != 2:
        raise ValueError("taxa_programs must be a 2D array")
    if activations.shape[1] != taxa_programs.shape[0]:
        raise ValueError("activation width must match number of taxa programs")
    if dead_threshold < 0:
        raise ValueError("dead_threshold must be non-negative")

    usage = activations.mean(axis=0)
    activation_entropy = -(activations * np.log(np.clip(activations, 1e-12, None))).sum(axis=1)
    dictionary_entropy = -(taxa_programs * np.log(np.clip(taxa_programs, 1e-12, None))).sum(axis=1)
    norms = np.linalg.norm(taxa_programs, axis=1, keepdims=True)
    normalized_programs = np.divide(
        taxa_programs,
        norms,
        out=np.zeros_like(taxa_programs),
        where=norms > 0,
    )
    cosine = normalized_programs @ normalized_programs.T
    if cosine.shape[0] > 1:
        offdiag = cosine[~np.eye(cosine.shape[0], dtype=bool)]
        cosine_mean = float(offdiag.mean())
        cosine_max = float(offdiag.max())
        cosine_p90 = float(np.quantile(offdiag, 0.90))
        cosine_p95 = float(np.quantile(offdiag, 0.95))
    else:
        cosine_mean = 0.0
        cosine_max = 0.0
        cosine_p90 = 0.0
        cosine_p95 = 0.0

    top_k = min(20, taxa_programs.shape[1])
    top_sets: list[set[int]] = []
    for weights in taxa_programs:
        order = np.argsort(-weights, kind="mergesort")[:top_k]
        top_sets.append(set(int(idx) for idx in order))
    overlaps: list[float] = []
    jaccards: list[float] = []
    for left_idx in range(len(top_sets)):
        for right_idx in range(left_idx + 1, len(top_sets)):
            intersection = len(top_sets[left_idx] & top_sets[right_idx])
            union = len(top_sets[left_idx] | top_sets[right_idx])
            overlaps.append(float(intersection))
            jaccards.append(float(intersection / union) if union > 0 else 0.0)
    top20_overlap_mean = float(np.mean(overlaps)) if overlaps else 0.0
    top20_overlap_max = float(np.max(overlaps)) if overlaps else 0.0
    top20_jaccard_mean = float(np.mean(jaccards)) if jaccards else 0.0

    rows: list[dict[str, object]] = [
        {"program": "all", "metric": "dead_program_fraction", "value": round(float((usage < dead_threshold).mean()), 6)},
        {"program": "all", "metric": "effective_programs_mean", "value": round(float(np.exp(activation_entropy).mean()), 6)},
        {"program": "all", "metric": "activation_entropy_mean", "value": round(float(activation_entropy.mean()), 6)},
        {"program": "all", "metric": "dictionary_entropy_mean", "value": round(float(dictionary_entropy.mean()), 6)},
        {"program": "all", "metric": "dictionary_cosine_mean_offdiag", "value": round(cosine_mean, 6)},
        {"program": "all", "metric": "dictionary_cosine_max_offdiag", "value": round(cosine_max, 6)},
        {"program": "all", "metric": "dictionary_cosine_p90_offdiag", "value": round(cosine_p90, 6)},
        {"program": "all", "metric": "dictionary_cosine_p95_offdiag", "value": round(cosine_p95, 6)},
        {"program": "all", "metric": "dictionary_top20_overlap_mean", "value": round(top20_overlap_mean, 6)},
        {"program": "all", "metric": "dictionary_top20_overlap_max", "value": round(top20_overlap_max, 6)},
        {"program": "all", "metric": "dictionary_top20_jaccard_mean", "value": round(top20_jaccard_mean, 6)},
    ]
    for program_idx, value in enumerate(usage):
        rows.append(
            {
                "program": int(program_idx),
                "metric": "program_mean_usage",
                "value": round(float(value), 6),
            }
        )
    return rows


def biome_enrichment_rows(
    activations: np.ndarray,
    metadata: pd.DataFrame,
    label_column: str,
) -> list[dict[str, object]]:
    """Summarize mean program activation by metadata label."""
    if activations.ndim != 2:
        raise ValueError("activations must be a 2D array")
    if len(metadata) != activations.shape[0]:
        raise ValueError("metadata rows must match activation rows")
    if label_column not in metadata.columns:
        raise ValueError(f"metadata must contain {label_column!r}")

    labels = metadata[label_column].astype("string")
    rows: list[dict[str, object]] = []
    for program_idx in range(activations.shape[1]):
        label_stats = []
        for label in sorted(labels.dropna().unique()):
            mask = labels == label
            values = activations[mask.to_numpy(dtype=bool), program_idx]
            label_stats.append((str(label), float(values.mean()), int(mask.sum())))
        label_stats.sort(key=lambda item: (-item[1], item[0]))
        for label, mean_activation, count in label_stats:
            rows.append(
                {
                    "program": int(program_idx),
                    "label": label,
                    "mean_activation": round(mean_activation, 6),
                    "num_samples": count,
                }
            )
    return rows


def write_json(path: Path, payload: dict[str, object]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
