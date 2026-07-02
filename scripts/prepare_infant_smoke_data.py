#!/usr/bin/env python3
"""Prepare a small deterministic infant-data subset for MGM smoke tests."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


DELIVERY_RE = re.compile(r"\(([CV])\)$")


def parse_delivery_label(env_label: str) -> str:
    """Return the C/V delivery suffix from labels such as ``12M(C)``."""
    match = DELIVERY_RE.search(env_label.strip())
    if not match:
        raise ValueError(f"Cannot parse C/V delivery suffix from Env label: {env_label!r}")
    return match.group(1)


def _read_metadata(path: Path, label_column: str) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "SampleID" not in reader.fieldnames or label_column not in reader.fieldnames:
            raise ValueError(
                f"{path} must contain SampleID and {label_column!r} columns; "
                f"found {reader.fieldnames}"
            )
        rows = []
        for row in reader:
            env = row[label_column]
            rows.append(
                {
                    "SampleID": row["SampleID"],
                    label_column: env,
                    "delivery_cv": parse_delivery_label(env),
                }
            )
    return rows


def _read_abundance_header(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{path} is empty") from exc
    if len(header) < 2:
        raise ValueError(f"{path} must have a feature-name column and at least one sample column")
    return header


def select_balanced_samples(
    metadata_rows: Iterable[dict[str, str]],
    available_sample_ids: set[str],
    max_samples: int,
    seed: int,
) -> list[dict[str, str]]:
    """Select a deterministic approximately balanced C/V sample subset."""
    eligible = [row for row in metadata_rows if row["SampleID"] in available_sample_ids]
    if not eligible:
        raise ValueError("No metadata SampleID values overlap abundance sample columns")
    if max_samples <= 0 or max_samples >= len(eligible):
        return sorted(eligible, key=lambda row: row["SampleID"])

    rng = random.Random(seed)
    by_delivery: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in eligible:
        by_delivery[row["delivery_cv"]].append(row)
    for rows in by_delivery.values():
        rng.shuffle(rows)

    labels = sorted(by_delivery)
    selected: list[dict[str, str]] = []
    while len(selected) < max_samples and any(by_delivery.values()):
        for label in labels:
            if len(selected) >= max_samples:
                break
            if by_delivery[label]:
                selected.append(by_delivery[label].pop())

    return selected


def _write_abundance_subset(input_path: Path, output_path: Path, selected_sample_ids: list[str]) -> None:
    selected = set(selected_sample_ids)
    with input_path.open(newline="") as source, output_path.open("w", newline="") as target:
        reader = csv.reader(source)
        writer = csv.writer(target)
        header = next(reader)
        index_by_sample = {sample_id: idx for idx, sample_id in enumerate(header)}
        missing = [sample_id for sample_id in selected_sample_ids if sample_id not in index_by_sample]
        if missing:
            raise ValueError(f"Selected sample IDs missing from abundance header: {missing[:5]}")

        keep_indices = [0] + [index_by_sample[sample_id] for sample_id in selected_sample_ids]
        writer.writerow([header[idx] for idx in keep_indices])
        for row in reader:
            if not row:
                continue
            writer.writerow([row[idx] for idx in keep_indices])

    if not selected:
        raise ValueError("No samples selected for abundance subset")


def _write_label_csv(path: Path, sample_rows: list[dict[str, str]], column_name: str, source_key: str) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["SampleID", column_name])
        for row in sample_rows:
            writer.writerow([row["SampleID"], row[source_key]])


def prepare_infant_smoke_data(
    abundance_path: Path,
    metadata_path: Path,
    output_dir: Path,
    max_samples: int,
    seed: int,
    label_column: str = "Env",
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    header = _read_abundance_header(abundance_path)
    sample_ids = header[1:]
    metadata_rows = _read_metadata(metadata_path, label_column)
    selected_rows = select_balanced_samples(metadata_rows, set(sample_ids), max_samples, seed)
    selected_rows = sorted(selected_rows, key=lambda row: row["SampleID"])
    selected_sample_ids = [row["SampleID"] for row in selected_rows]

    _write_abundance_subset(abundance_path, output_dir / "abundance.csv", selected_sample_ids)
    _write_label_csv(output_dir / "labels_env.csv", selected_rows, "Env", label_column)
    _write_label_csv(output_dir / "labels_delivery_cv.csv", selected_rows, "delivery_cv", "delivery_cv")

    manifest = {
        "abundance_input": str(abundance_path),
        "metadata_input": str(metadata_path),
        "max_samples": max_samples,
        "seed": seed,
        "num_available_samples": len(sample_ids),
        "num_selected_samples": len(selected_sample_ids),
        "selected_sample_ids": selected_sample_ids,
        "env_counts": dict(sorted(Counter(row[label_column] for row in selected_rows).items())),
        "delivery_counts": dict(sorted(Counter(row["delivery_cv"] for row in selected_rows).items())),
        "outputs": {
            "abundance": str(output_dir / "abundance.csv"),
            "labels_env": str(output_dir / "labels_env.csv"),
            "labels_delivery_cv": str(output_dir / "labels_delivery_cv.csv"),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abundance", type=Path, default=Path("infant_data/abundance.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("infant_data/meta_withbirth.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/mgm_repro/infant_smoke/data"))
    parser.add_argument("--max-samples", type=int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label-column", default="Env")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = prepare_infant_smoke_data(
        abundance_path=args.abundance,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        seed=args.seed,
        label_column=args.label_column,
    )
    print(
        "Prepared infant smoke data: "
        f"{manifest['num_selected_samples']} samples -> {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
