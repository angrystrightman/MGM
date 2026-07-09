#!/usr/bin/env python3
"""Summarize ComPASS-Biome NMF warm-start bottleneck runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from scripts.compass_biome_utils import program_diagnostics_rows


def parse_run_arg(value: str) -> tuple[str, Path]:
    """Parse a LABEL=PATH argument."""
    if "=" not in value:
        raise argparse.ArgumentTypeError("run arguments must be formatted as LABEL=PATH")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("run label must not be empty")
    return label, Path(path)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _test_metrics(path: Path) -> dict[str, float]:
    metrics = pd.read_csv(path)
    test_metrics = metrics.loc[metrics["split"].astype(str).eq("test")].copy()
    rows: dict[str, float] = {}
    for model in ["real", "nmf_oracle", "mean_baseline", "shuffle"]:
        model_rows = test_metrics.loc[test_metrics["model"].astype(str).eq(model)]
        if model_rows.empty:
            continue
        row = model_rows.iloc[0]
        prefix = f"test_{model}"
        rows[f"{prefix}_cross_entropy"] = float(row["cross_entropy"])
        rows[f"{prefix}_top20_recall"] = float(row["top_20_recall"])
        rows[f"{prefix}_bray_curtis"] = float(row["bray_curtis_similarity"])
    real_prefix = "test_real" if "test_real_top20_recall" in rows else "test_nmf_oracle"
    if f"{real_prefix}_top20_recall" in rows:
        top20_controls = [
            rows[key]
            for key in ["test_mean_baseline_top20_recall", "test_shuffle_top20_recall"]
            if key in rows
        ]
        rows["test_top20_control_gap"] = (
            float(rows[f"{real_prefix}_top20_recall"] - max(top20_controls))
            if top20_controls
            else float("nan")
        )
    if f"{real_prefix}_bray_curtis" in rows:
        bray_controls = [
            rows[key]
            for key in ["test_mean_baseline_bray_curtis", "test_shuffle_bray_curtis"]
            if key in rows
        ]
        rows["test_bray_control_gap"] = (
            float(rows[f"{real_prefix}_bray_curtis"] - max(bray_controls))
            if bray_controls
            else float("nan")
        )
    return rows


def _diagnostics_from_csv(path: Path) -> dict[str, float]:
    diagnostics = pd.read_csv(path)
    all_rows = diagnostics.loc[diagnostics["program"].astype(str).eq("all")]
    return {str(row["metric"]): float(row["value"]) for _, row in all_rows.iterrows()}


def _dictionary_diagnostics(taxa_programs: np.ndarray) -> dict[str, float]:
    activations = np.full((1, taxa_programs.shape[0]), 1.0 / taxa_programs.shape[0], dtype=np.float32)
    diagnostics = {
        str(row["metric"]): float(row["value"])
        for row in program_diagnostics_rows(activations, taxa_programs)
        if str(row["program"]) == "all"
    }
    for key in ["dead_program_fraction", "effective_programs_mean", "activation_entropy_mean"]:
        diagnostics[key] = float("nan")
    return diagnostics


def _dictionary_drift(init: np.ndarray, final: np.ndarray, eps: float = 1e-8) -> dict[str, float]:
    init = np.asarray(init, dtype=np.float32)
    final = np.asarray(final, dtype=np.float32)
    if init.shape != final.shape:
        raise ValueError(f"dictionary init shape {init.shape} does not match final shape {final.shape}")
    init = init / init.sum(axis=1, keepdims=True).clip(min=eps)
    final = final / final.sum(axis=1, keepdims=True).clip(min=eps)
    kl = (init.clip(min=eps) * (np.log(init.clip(min=eps)) - np.log(final.clip(min=eps)))).sum(axis=1)
    return {
        "nmf_init_to_final_kl_mean": float(kl.mean()),
        "nmf_init_to_final_kl_max": float(kl.max()),
        "nmf_init_to_final_max_abs_diff": float(np.max(np.abs(init - final))),
    }


def _final_training_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    training = pd.read_csv(path)
    if training.empty:
        return {}
    final = training.iloc[-1]
    rows: dict[str, float] = {}
    for key in [
        "regularized_loss",
        "reconstruction_loss",
        "dictionary_diversity_loss",
        "dictionary_anchor_loss",
        "sample_entropy_mean",
        "usage_balance_loss",
    ]:
        if key in final:
            rows[f"final_{key}"] = float(final[key])
    return rows


def _mode_from_manifest(label: str, manifest: dict[str, object]) -> str:
    if bool(manifest.get("freeze_taxa_dictionary", False)):
        return "nmf_fixed"
    if float(manifest.get("dictionary_anchor_weight", 0.0)) > 0:
        return "nmf_anchor"
    if manifest.get("taxa_dictionary_init"):
        return "nmf_trainable"
    return label


def summarize_bottleneck_run(
    label: str,
    output_dir: Path,
    dictionary_init: np.ndarray | None = None,
) -> dict[str, object]:
    """Summarize one bottleneck output directory."""
    manifest = _read_json(output_dir / "manifest.json")
    row: dict[str, object] = {
        "run": label,
        "mode": _mode_from_manifest(label, manifest),
        "output_dir": str(output_dir),
        "num_samples": int(manifest["num_samples"]),
        "num_taxa": int(manifest["num_taxa"]),
        "embedding_width": int(manifest["embedding_width"]),
        "num_programs": int(manifest["num_programs"]),
        "epochs_ran": int(manifest["epochs_ran"]),
        "batch_size": int(manifest["batch_size"]),
        "learning_rate": float(manifest["learning_rate"]),
        "usage_balance_weight": float(manifest.get("usage_balance_weight", 0.0)),
        "sample_entropy_weight": float(manifest.get("sample_entropy_weight", 0.0)),
        "dictionary_diversity_weight": float(manifest.get("dictionary_diversity_weight", 0.0)),
        "dictionary_anchor_weight": float(manifest.get("dictionary_anchor_weight", 0.0)),
        "freeze_taxa_dictionary": bool(manifest.get("freeze_taxa_dictionary", False)),
    }
    row.update(_test_metrics(output_dir / "reconstruction_metrics.csv"))
    row.update(_diagnostics_from_csv(output_dir / "program_diagnostics.csv"))
    row.update(_final_training_metrics(output_dir / "training_metrics.csv"))
    if dictionary_init is not None:
        with np.load(output_dir / "bottleneck_outputs.npz", allow_pickle=False) as payload:
            row.update(_dictionary_drift(dictionary_init, payload["taxa_programs"]))
    return row


def summarize_nmf_oracle(label: str, nmf_dir: Path) -> dict[str, object]:
    """Summarize the train-split-only NMF oracle report."""
    manifest = _read_json(nmf_dir / "manifest.json")
    with np.load(nmf_dir / "nmf_dictionary.npz", allow_pickle=False) as payload:
        taxa_programs = payload["taxa_programs"]
    row: dict[str, object] = {
        "run": label,
        "mode": "nmf_oracle",
        "output_dir": str(nmf_dir),
        "num_samples": int(manifest["num_samples"]),
        "num_taxa": int(manifest["num_taxa"]),
        "embedding_width": float("nan"),
        "num_programs": int(manifest["num_programs"]),
        "epochs_ran": int(manifest["n_iter"]),
        "batch_size": float("nan"),
        "learning_rate": float("nan"),
        "usage_balance_weight": float("nan"),
        "sample_entropy_weight": float("nan"),
        "dictionary_diversity_weight": float("nan"),
        "dictionary_anchor_weight": float("nan"),
        "freeze_taxa_dictionary": False,
        "nmf_reconstruction_err": float(manifest["reconstruction_err"]),
    }
    row.update(_test_metrics(nmf_dir / "nmf_oracle_metrics.csv"))
    row.update(_dictionary_diagnostics(taxa_programs))
    row.update(
        {
            "nmf_init_to_final_kl_mean": 0.0,
            "nmf_init_to_final_kl_max": 0.0,
            "nmf_init_to_final_max_abs_diff": 0.0,
        }
    )
    return row


def recommend_nmf_warm_start(summary: pd.DataFrame, parent_label: str) -> dict[str, object]:
    """Return a conservative recommendation for NMF warm-start modes."""
    if summary.empty:
        raise ValueError("summary must not be empty")
    parent_rows = summary.loc[summary["run"].astype(str).eq(parent_label)]
    if parent_rows.empty:
        raise ValueError(f"parent run {parent_label!r} not found")
    parent = parent_rows.iloc[0]
    candidate_mask = summary["mode"].astype(str).isin(["nmf_fixed", "nmf_trainable", "nmf_anchor"])
    candidates = summary.loc[candidate_mask].copy()
    if candidates.empty:
        raise ValueError("summary must include at least one NMF bottleneck candidate")

    candidates["beats_parent_top20"] = candidates["test_real_top20_recall"] >= float(parent["test_real_top20_recall"])
    candidates["beats_parent_bray"] = candidates["test_real_bray_curtis"] >= float(parent["test_real_bray_curtis"])
    candidates["passes_usage_dictionary_gate"] = (
        (candidates["dead_program_fraction"] <= 0.25)
        & (candidates["dictionary_cosine_p95_offdiag"] <= 0.20)
    )
    candidates["keeps_control_gap"] = (
        (candidates["test_top20_control_gap"] >= 0.10)
        & (candidates["test_bray_control_gap"] >= 0.10)
    )
    positive = candidates.loc[
        candidates["beats_parent_top20"]
        & candidates["beats_parent_bray"]
        & candidates["passes_usage_dictionary_gate"]
        & candidates["keeps_control_gap"]
    ]
    if positive.empty:
        best = candidates.sort_values(
            ["test_real_bray_curtis", "test_real_top20_recall"],
            ascending=[False, False],
        ).iloc[0]
        verdict = "tradeoff"
        reason = "No NMF mode beat the parent on both Top-20 recall and Bray-Curtis; best mode improves one axis."
    else:
        best = positive.sort_values(
            ["test_real_bray_curtis", "test_real_top20_recall"],
            ascending=[False, False],
        ).iloc[0]
        verdict = "positive"
        reason = "At least one NMF mode beat the parent on both reconstruction metrics while passing usage/dictionary gates."

    fixed_rows = candidates.loc[candidates["mode"].astype(str).eq("nmf_fixed")]
    fixed_interpretability_positive = False
    if not fixed_rows.empty:
        fixed = fixed_rows.iloc[0]
        fixed_interpretability_positive = bool(
            fixed["test_top20_control_gap"] >= 0.10
            and fixed["test_bray_control_gap"] >= 0.10
            and fixed["dead_program_fraction"] <= 0.25
            and fixed["dictionary_cosine_p95_offdiag"] <= 0.20
        )

    return {
        "parent_run": parent_label,
        "verdict": verdict,
        "recommended_mode": str(best["mode"]),
        "recommended_run": str(best["run"]),
        "reason": reason,
        "fixed_interpretability_positive": fixed_interpretability_positive,
        "parent_test_top20_recall": float(parent["test_real_top20_recall"]),
        "parent_test_bray_curtis": float(parent["test_real_bray_curtis"]),
        "recommended_test_top20_recall": float(best["test_real_top20_recall"]),
        "recommended_test_bray_curtis": float(best["test_real_bray_curtis"]),
        "recommended_dead_program_fraction": float(best["dead_program_fraction"]),
        "recommended_dictionary_cosine_p95_offdiag": float(best["dictionary_cosine_p95_offdiag"]),
        "notes": [
            "Positive requires beating the parent on both Top-20 recall and Bray-Curtis.",
            "Tradeoff can still be useful when Bray-Curtis and dictionary interpretability improve while Top-20 recall falls.",
        ],
    }


def summarize_nmf_warm_start_runs(
    parent_run: tuple[str, Path],
    nmf_dir: Path,
    runs: Iterable[tuple[str, Path]],
    output_summary: Path,
    output_recommendation: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Write comparison summary and recommendation files."""
    with np.load(nmf_dir / "nmf_dictionary.npz", allow_pickle=False) as payload:
        dictionary_init = payload["taxa_programs"].astype(np.float32)

    parent_label, parent_path = parent_run
    rows = [
        summarize_bottleneck_run(parent_label, parent_path, dictionary_init=None),
        summarize_nmf_oracle("NMF_oracle", nmf_dir),
    ]
    for label, path in runs:
        rows.append(summarize_bottleneck_run(label, path, dictionary_init=dictionary_init))

    summary = pd.DataFrame(rows)
    recommendation = recommend_nmf_warm_start(summary, parent_label=parent_label)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_summary, index=False)
    output_recommendation.write_text(json.dumps(recommendation, indent=2, sort_keys=True) + "\n")
    return summary, recommendation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", type=parse_run_arg, required=True)
    parser.add_argument("--nmf-dir", type=Path, required=True)
    parser.add_argument("--run", type=parse_run_arg, action="append", default=[])
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--output-recommendation", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary, recommendation = summarize_nmf_warm_start_runs(
        parent_run=args.parent_run,
        nmf_dir=args.nmf_dir,
        runs=args.run,
        output_summary=args.output_summary,
        output_recommendation=args.output_recommendation,
    )
    print(summary.to_string(index=False))
    print(json.dumps(recommendation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
