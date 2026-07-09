#!/usr/bin/env python3
"""Summarize regularized ComPASS-Biome K-resolution bottleneck runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from scripts.compass_biome_utils import program_diagnostics_rows


def parse_run_arg(value: str) -> tuple[str, Path]:
    """Parse a LABEL=PATH run argument."""
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be formatted as LABEL=PATH")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("run label must not be empty")
    return label, Path(path)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _test_metric_rows(metrics: pd.DataFrame) -> dict[str, float]:
    rows: dict[str, float] = {}
    test_metrics = metrics.loc[metrics["split"].astype(str).eq("test")].copy()
    for model_name in ["real", "mean_baseline", "shuffle"]:
        model_rows = test_metrics.loc[test_metrics["model"].astype(str).eq(model_name)]
        if model_rows.empty:
            continue
        row = model_rows.iloc[0]
        prefix = f"test_{model_name}"
        rows[f"{prefix}_cross_entropy"] = float(row["cross_entropy"])
        rows[f"{prefix}_top20_recall"] = float(row["top_20_recall"])
        rows[f"{prefix}_bray_curtis"] = float(row["bray_curtis_similarity"])
    if "test_real_top20_recall" in rows:
        control_top20 = max(
            rows.get("test_mean_baseline_top20_recall", float("-inf")),
            rows.get("test_shuffle_top20_recall", float("-inf")),
        )
        rows["test_top20_control_gap"] = float(rows["test_real_top20_recall"] - control_top20)
    if "test_real_bray_curtis" in rows:
        control_bray = max(
            rows.get("test_mean_baseline_bray_curtis", float("-inf")),
            rows.get("test_shuffle_bray_curtis", float("-inf")),
        )
        rows["test_bray_control_gap"] = float(rows["test_real_bray_curtis"] - control_bray)
    return rows


def _diagnostic_map(activations: np.ndarray, taxa_programs: np.ndarray) -> dict[str, float]:
    diagnostics = program_diagnostics_rows(activations, taxa_programs)
    return {
        str(row["metric"]): float(row["value"])
        for row in diagnostics
        if str(row["program"]) == "all"
    }


def _program_usage_map(activations: np.ndarray, taxa_programs: np.ndarray) -> dict[int, float]:
    diagnostics = program_diagnostics_rows(activations, taxa_programs)
    return {
        int(row["program"]): float(row["value"])
        for row in diagnostics
        if row["metric"] == "program_mean_usage"
    }


def _top_taxa_by_program(path: Path, top_n: int = 5) -> dict[int, str]:
    if not path.exists():
        return {}
    taxa = pd.read_csv(path)
    taxa = taxa.loc[taxa["rank"].astype(int) <= top_n].copy()
    grouped: dict[int, str] = {}
    for program, group in taxa.groupby("program", sort=True):
        ordered = group.sort_values("rank")
        grouped[int(program)] = ";".join(ordered["taxon"].astype(str).tolist())
    return grouped


def _top_biome_by_program(path: Path) -> dict[int, dict[str, object]]:
    if not path.exists():
        return {}
    enrichment = pd.read_csv(path)
    rows: dict[int, dict[str, object]] = {}
    for program, group in enrichment.groupby("program", sort=True):
        ordered = group.sort_values(["mean_activation", "label"], ascending=[False, True])
        top = ordered.iloc[0]
        mean_activation = float(group["mean_activation"].mean())
        lift = float(top["mean_activation"] / mean_activation) if mean_activation > 0 else float("nan")
        rows[int(program)] = {
            "top_biome_label": str(top["label"]),
            "top_biome_mean_activation": float(top["mean_activation"]),
            "top_biome_num_samples": int(top["num_samples"]),
            "top_biome_activation_lift": lift,
        }
    return rows


def summarize_single_run(label: str, output_dir: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Return one run-level summary row and per-program report rows."""
    manifest_path = output_dir / "manifest.json"
    metrics_path = output_dir / "reconstruction_metrics.csv"
    outputs_path = output_dir / "bottleneck_outputs.npz"
    training_path = output_dir / "training_metrics.csv"
    for path in [manifest_path, metrics_path, outputs_path, training_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required K-resolution artifact missing: {path}")

    manifest = _read_json(manifest_path)
    metrics = pd.read_csv(metrics_path)
    training = pd.read_csv(training_path)
    final_training = training.iloc[-1].to_dict()
    with np.load(outputs_path, allow_pickle=False) as payload:
        activations = payload["activations"]
        taxa_programs = payload["taxa_programs"]

    diagnostics = _diagnostic_map(activations, taxa_programs)
    summary: dict[str, object] = {
        "run": label,
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
        "sample_entropy_target_effective": float(manifest.get("sample_entropy_target_effective", 2.0)),
        "dictionary_diversity_weight": float(manifest.get("dictionary_diversity_weight", 0.0)),
        "dictionary_diversity_loss_type": str(manifest.get("dictionary_diversity_loss_type", "none")),
        "dictionary_diversity_threshold": float(manifest.get("dictionary_diversity_threshold", 0.0)),
        "final_regularized_loss": float(final_training.get("regularized_loss", final_training.get("loss", np.nan))),
        "final_reconstruction_loss": float(final_training.get("reconstruction_loss", np.nan)),
        "final_dictionary_diversity_loss": float(final_training.get("dictionary_diversity_loss", np.nan)),
    }
    summary.update(_test_metric_rows(metrics))
    summary.update(diagnostics)

    usage = _program_usage_map(activations, taxa_programs)
    top_taxa = _top_taxa_by_program(output_dir / "top_taxa_per_program.csv")
    top_biome = _top_biome_by_program(output_dir / "biome_enrichment.csv")
    program_rows: list[dict[str, object]] = []
    for program_idx in range(int(manifest["num_programs"])):
        biome_row = top_biome.get(program_idx, {})
        program_rows.append(
            {
                "run": label,
                "num_programs": int(manifest["num_programs"]),
                "program": program_idx,
                "program_mean_usage": usage.get(program_idx, float("nan")),
                "is_active": bool(usage.get(program_idx, 0.0) >= 0.01),
                "top_taxa": top_taxa.get(program_idx, ""),
                "top_biome_label": biome_row.get("top_biome_label", ""),
                "top_biome_mean_activation": biome_row.get("top_biome_mean_activation", float("nan")),
                "top_biome_num_samples": biome_row.get("top_biome_num_samples", 0),
                "top_biome_activation_lift": biome_row.get("top_biome_activation_lift", float("nan")),
            }
        )
    return summary, program_rows


def _reference_k32(summary: pd.DataFrame) -> pd.Series:
    k32 = summary.loc[summary["num_programs"].astype(int).eq(32)]
    if not k32.empty:
        return k32.iloc[0]
    return summary.sort_values(["test_real_bray_curtis", "test_real_top20_recall"], ascending=[False, False]).iloc[0]


def _eligible_high_fidelity(summary: pd.DataFrame, reference: pd.Series) -> pd.DataFrame:
    return summary.loc[
        (summary["test_top20_control_gap"] >= 0.10)
        & (summary["test_bray_control_gap"] >= 0.10)
        & (summary["test_real_top20_recall"] >= 0.90 * float(reference["test_real_top20_recall"]))
        & (summary["test_real_bray_curtis"] >= 0.90 * float(reference["test_real_bray_curtis"]))
        & (summary["dead_program_fraction"] <= 0.25)
        & (summary["dictionary_cosine_p95_offdiag"] <= 0.20)
    ]


def _eligible_compact(summary: pd.DataFrame, reference: pd.Series) -> pd.DataFrame:
    return summary.loc[
        (summary["test_top20_control_gap"] >= 0.10)
        & (summary["test_bray_control_gap"] >= 0.10)
        & (summary["test_real_top20_recall"] >= 0.80 * float(reference["test_real_top20_recall"]))
        & (summary["test_real_bray_curtis"] >= 0.80 * float(reference["test_real_bray_curtis"]))
    ]


def recommend_k_resolution(summary: pd.DataFrame) -> dict[str, object]:
    """Recommend primary, compact, and high-resolution K from a summary table."""
    if summary.empty:
        raise ValueError("summary must contain at least one run")
    summary = summary.copy()
    summary["num_programs"] = summary["num_programs"].astype(int)
    if "test_top20_control_gap" not in summary.columns:
        top20_controls = [
            col
            for col in ["test_mean_baseline_top20_recall", "test_shuffle_top20_recall"]
            if col in summary.columns
        ]
        if not top20_controls:
            raise ValueError("summary must contain at least one Top-20 control column")
        summary["test_top20_control_gap"] = summary["test_real_top20_recall"] - summary[top20_controls].max(axis=1)
    if "test_bray_control_gap" not in summary.columns:
        bray_controls = [
            col
            for col in ["test_mean_baseline_bray_curtis", "test_shuffle_bray_curtis"]
            if col in summary.columns
        ]
        if not bray_controls:
            raise ValueError("summary must contain at least one Bray-Curtis control column")
        summary["test_bray_control_gap"] = summary["test_real_bray_curtis"] - summary[bray_controls].max(axis=1)
    reference = _reference_k32(summary)

    high_fidelity = _eligible_high_fidelity(summary, reference)
    if high_fidelity.empty:
        primary_row = reference
        primary_reason = "No smaller K met all high-fidelity gates; using reference/best reconstruction run."
    else:
        primary_row = high_fidelity.sort_values(["num_programs", "test_real_bray_curtis"], ascending=[True, False]).iloc[0]
        primary_reason = "Smallest K meeting reconstruction, control-gap, usage, and dictionary gates."

    compact = _eligible_compact(summary, reference)
    compact_row = (
        primary_row
        if compact.empty
        else compact.sort_values(["num_programs", "test_real_bray_curtis"], ascending=[True, False]).iloc[0]
    )

    prefer_k64 = False
    high_resolution_k = int(reference["num_programs"])
    k64 = summary.loc[summary["num_programs"].eq(64)]
    k32 = summary.loc[summary["num_programs"].eq(32)]
    if not k64.empty and not k32.empty:
        k64_row = k64.iloc[0]
        k32_row = k32.iloc[0]
        top20_gain = float(k64_row["test_real_top20_recall"]) / float(k32_row["test_real_top20_recall"]) - 1.0
        bray_gain = float(k64_row["test_real_bray_curtis"]) / float(k32_row["test_real_bray_curtis"]) - 1.0
        no_worse_usage = float(k64_row["dead_program_fraction"]) <= float(k32_row["dead_program_fraction"])
        no_worse_dictionary = float(k64_row["dictionary_cosine_p95_offdiag"]) <= float(k32_row["dictionary_cosine_p95_offdiag"])
        prefer_k64 = max(top20_gain, bray_gain) >= 0.02 and no_worse_usage and no_worse_dictionary
        high_resolution_k = 64 if prefer_k64 else 32

    rejected = summary.loc[
        (summary["test_top20_control_gap"] < 0.10)
        | (summary["test_bray_control_gap"] < 0.10),
        "run",
    ].astype(str).tolist()

    return {
        "primary_k": int(primary_row["num_programs"]),
        "primary_run": str(primary_row["run"]),
        "primary_reason": primary_reason,
        "compact_k": int(compact_row["num_programs"]),
        "compact_run": str(compact_row["run"]),
        "high_resolution_k": int(high_resolution_k),
        "prefer_k64_over_k32": bool(prefer_k64),
        "reference_run": str(reference["run"]),
        "rejected_runs": rejected,
        "criteria": {
            "control_gap_minimum": 0.10,
            "primary_reconstruction_fraction_of_reference": 0.90,
            "compact_reconstruction_fraction_of_reference": 0.80,
            "primary_dead_program_fraction_max": 0.25,
            "primary_dictionary_p95_cosine_max": 0.20,
            "k64_minimum_relative_gain": 0.02,
        },
    }


def summarize_k_resolution_runs(
    runs: Mapping[str, Path],
    output_summary: Path,
    output_program_report: Path,
    output_recommendation: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Write run-level, program-level, and recommendation outputs for K sweep runs."""
    summary_rows: list[dict[str, object]] = []
    program_rows: list[dict[str, object]] = []
    for label, path in runs.items():
        summary, programs = summarize_single_run(label, Path(path))
        summary_rows.append(summary)
        program_rows.extend(programs)

    summary_df = pd.DataFrame(summary_rows).sort_values("num_programs").reset_index(drop=True)
    program_df = pd.DataFrame(program_rows).sort_values(["num_programs", "program"]).reset_index(drop=True)
    recommendation = recommend_k_resolution(summary_df)

    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_program_report.parent.mkdir(parents=True, exist_ok=True)
    output_recommendation.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_summary, index=False)
    program_df.to_csv(output_program_report, index=False)
    output_recommendation.write_text(json.dumps(recommendation, indent=2, sort_keys=True) + "\n")
    return summary_df, program_df, recommendation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run_arg, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--output-program-report", type=Path, required=True)
    parser.add_argument("--output-recommendation", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs = dict(args.run)
    summary, _, recommendation = summarize_k_resolution_runs(
        runs,
        output_summary=args.output_summary,
        output_program_report=args.output_program_report,
        output_recommendation=args.output_recommendation,
    )
    print(summary.to_string(index=False))
    print(json.dumps(recommendation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
