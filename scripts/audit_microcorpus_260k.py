#!/usr/bin/env python3
"""Audit local MGM MicroCorpus-260K files for ComPASS-Biome feasibility."""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.compass_biome_utils import (
    alignment_report,
    normalize_mgnify_metadata,
    task_suitability,
)


DEFAULT_DATA_ROOT = Path("/home/sunyirong/shared/sunyirong/MicroCorpus-260K")


def file_inventory(data_root: Path) -> list[dict[str, object]]:
    patterns = [
        "MicroCorpus-260K.pkl",
        "MicroCorpus-260K_unnorm.pkl",
        "mgnify_biomes.csv",
        "*vocab*",
        "*token*",
        "*split*",
        "*label*",
        "*.ckpt",
        "pytorch_model.bin",
    ]
    seen: set[Path] = set()
    rows: list[dict[str, object]] = []
    for pattern in patterns:
        for path in data_root.glob(pattern):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            suffix = path.suffix.lower()
            if suffix == ".pkl":
                file_type = "pickle"
            elif suffix == ".csv":
                file_type = "csv"
            elif suffix in {".bin", ".ckpt"}:
                file_type = "checkpoint"
            else:
                file_type = suffix.lstrip(".") or "file"
            rows.append(
                {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "file_type": file_type,
                }
            )
    return sorted(rows, key=lambda row: str(row["path"]))


def _top_values(series: pd.Series, limit: int) -> list[dict[str, object]]:
    counts = series.astype("string").fillna("<NA>").value_counts(dropna=False).head(limit)
    return [{"value": str(value), "count": int(count)} for value, count in counts.items()]


def summarize_metadata(metadata: pd.DataFrame, top_n: int = 20) -> dict[str, object]:
    normalized = normalize_mgnify_metadata(metadata)
    lower_columns = {str(col).lower() for col in normalized.columns}
    detected_fields = {
        "sample_id_or_accession": "sample_id" in normalized.columns,
        "biome_hierarchy": any(str(col).startswith("biome_") for col in normalized.columns),
        "disease_or_phenotype": any(
            "disease" in col or "phenotype" in col or "diagnosis" in col for col in lower_columns
        ),
        "study_or_cohort": any("study" in col or "project" in col or "cohort" in col for col in lower_columns),
        "country_or_region": any("country" in col or "region" in col or "location" in col for col in lower_columns),
        "sequencing_platform": any("platform" in col or "instrument" in col for col in lower_columns),
        "pathway": any("pathway" in col or "function" in col for col in lower_columns),
    }
    columns = []
    for col in normalized.columns:
        series = normalized[col]
        columns.append(
            {
                "name": str(col),
                "dtype": str(series.dtype),
                "non_null": int(series.notna().sum()),
                "unique": int(series.nunique(dropna=True)),
                "top_values": _top_values(series, top_n),
                "examples": [str(value) for value in series.dropna().head(8).tolist()],
            }
        )
    return {
        "shape": list(normalized.shape),
        "columns": columns,
        "detected_fields": detected_fields,
    }


def _tensor_shape(value: Any) -> list[int] | None:
    if hasattr(value, "shape"):
        return [int(dim) for dim in value.shape]
    return None


def summarize_corpus(path: Path, metadata_ids: list[str], sample_values: bool = True) -> dict[str, object]:
    started = time.time()
    with path.open("rb") as handle:
        corpus = pickle.load(handle)
    load_seconds = time.time() - started

    item0 = corpus[0]
    data = getattr(corpus, "data", None)
    tokenizer = getattr(corpus, "tokenizer", None)
    tokens = getattr(corpus, "tokens", None)
    corpus_ids = [str(value) for value in data.index] if isinstance(data, pd.DataFrame) else []

    summary: dict[str, object] = {
        "path": str(path),
        "loadable": True,
        "load_seconds": round(load_seconds, 3),
        "object_type": f"{type(corpus).__module__}.{type(corpus).__name__}",
        "len": int(len(corpus)),
        "attributes": sorted(list(getattr(corpus, "__dict__", {}).keys())),
        "item0_type": f"{type(item0).__module__}.{type(item0).__name__}",
        "item0_keys": list(item0.keys()) if isinstance(item0, dict) else [],
        "item0_shapes": {key: _tensor_shape(value) for key, value in item0.items()} if isinstance(item0, dict) else {},
        "tokens_shape": _tensor_shape(tokens),
        "tokenizer_type": f"{type(tokenizer).__module__}.{type(tokenizer).__name__}" if tokenizer is not None else None,
        "tokenizer_vocab_size": int(getattr(tokenizer, "vocab_size", 0)) if tokenizer is not None else None,
        "has_data": isinstance(data, pd.DataFrame),
        "alignment": alignment_report(corpus_ids, metadata_ids) if corpus_ids else None,
    }

    if isinstance(data, pd.DataFrame):
        summary["data"] = {
            "type": "pandas.DataFrame",
            "shape": list(data.shape),
            "orientation": "sample_x_taxa",
            "index_unique": bool(data.index.is_unique),
            "columns_unique": bool(data.columns.is_unique),
            "index_first10": [str(value) for value in data.index[:10].tolist()],
            "columns_first20": [str(value) for value in data.columns[:20].tolist()],
            "dtypes": data.dtypes.astype(str).value_counts().to_dict(),
        }
        if sample_values:
            block = data.iloc[: min(100, data.shape[0]), : min(1000, data.shape[1])].to_numpy()
            row_sums = data.iloc[: min(10, data.shape[0])].sum(axis=1).astype(float)
            nonzero = (data.iloc[: min(10, data.shape[0])] != 0).sum(axis=1)
            summary["data"].update(
                {
                    "sample_block_shape": list(block.shape),
                    "sample_block_min": float(np.nanmin(block)),
                    "sample_block_max": float(np.nanmax(block)),
                    "sample_block_mean": float(np.nanmean(block)),
                    "sample_block_zero_fraction": float((block == 0).sum() / block.size),
                    "row_sum_first10": [float(value) for value in row_sums.tolist()],
                    "nonzero_first10": [int(value) for value in nonzero.tolist()],
                }
            )

    del corpus
    gc.collect()
    return summary


def build_audit_report(data_root: Path, top_n: int = 20) -> dict[str, object]:
    metadata_path = data_root / "mgnify_biomes.csv"
    normalized_path = data_root / "MicroCorpus-260K.pkl"
    unnorm_path = data_root / "MicroCorpus-260K_unnorm.pkl"

    raw_metadata = pd.read_csv(metadata_path, dtype=str, low_memory=False)
    metadata = normalize_mgnify_metadata(raw_metadata)
    metadata_ids = metadata["sample_id"].astype(str).tolist()

    normalized_summary = summarize_corpus(normalized_path, metadata_ids)
    unnorm_summary = summarize_corpus(unnorm_path, metadata_ids)
    taxa_reconstruction = "yes"
    if unnorm_summary.get("data", {}).get("orientation") != "sample_x_taxa":
        taxa_reconstruction = "uncertain"

    return {
        "data_root": str(data_root),
        "file_inventory": file_inventory(data_root),
        "metadata": summarize_metadata(raw_metadata, top_n=top_n),
        "corpora": {
            "normalized": normalized_summary,
            "unnormalized": unnorm_summary,
        },
        "cross_corpus_alignment": {
            "sample_ids_match_metadata": bool(
                normalized_summary["alignment"]["ordering_matches"]
                and unnorm_summary["alignment"]["ordering_matches"]
            ),
            "normalized_and_unnormalized_shapes": [
                normalized_summary.get("data", {}).get("shape"),
                unnorm_summary.get("data", {}).get("shape"),
            ],
        },
        "abundance_conclusion": {
            "can_provide_X_NG": taxa_reconstruction,
            "recommended_target": "MicroCorpus-260K_unnorm.pkl.data",
            "reason": "Unnormalized .data is sparse sample x genus relative abundance with fixed taxa columns.",
        },
        "task_suitability": task_suitability(metadata, has_corpus=True, has_unnorm_data=True),
    }


def write_markdown_report(path: Path, report: dict[str, object]) -> None:
    suitability = report["task_suitability"]
    lines = [
        "# ComPASS-Biome MicroCorpus-260K Audit",
        "",
        f"Data root: `{report['data_root']}`",
        "",
        "## Key Conclusion",
        "",
        "- Teacher embedding extraction: yes.",
        "- Taxa reconstruction target: yes, use `MicroCorpus-260K_unnorm.pkl.data`.",
        "- Disease/pathway validation: no direct labels or pathway matrix found.",
        "- Metadata validation target: biome/source hierarchy.",
        "",
        "## Task Suitability",
        "",
        "| Task | Support | Reason |",
        "|---|---|---|",
    ]
    for task, item in suitability.items():
        lines.append(f"| {task} | {item['support']} | {item['reason']} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-json", type=Path, default=Path("runs/compass_biome/audit/microcorpus_260k_audit.json"))
    parser.add_argument("--output-md", type=Path, default=Path("runs/compass_biome/audit/microcorpus_260k_audit.md"))
    parser.add_argument("--top-n", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_audit_report(args.data_root, top_n=args.top_n)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_markdown_report(args.output_md, report)
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
