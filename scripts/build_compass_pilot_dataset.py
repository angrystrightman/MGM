#!/usr/bin/env python3
"""Build an output-isolated ComPASS-Biome pilot dataset from MicroCorpus-260K."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_microcorpus_260k import DEFAULT_DATA_ROOT
from scripts.compass_biome_utils import (
    alignment_report,
    filter_and_normalize_taxa,
    normalize_mgnify_metadata,
    select_balanced_sample_ids,
    select_top_taxa_by_train,
    stratified_split_sample_ids,
)


def write_pilot_dataset(
    sample_ids: list[str],
    metadata: pd.DataFrame,
    teacher_ids: list[str],
    x_unnorm: pd.DataFrame,
    output_dir: Path,
    num_taxa: int | None = None,
    split_label_column: str = "biome_1",
    seed: int = 0,
) -> dict[str, object]:
    """Write aligned sample IDs, metadata, taxa names, and X target matrix."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_ids = [str(sample_id) for sample_id in sample_ids]
    metadata = normalize_mgnify_metadata(metadata)

    missing_teacher = sorted(set(sample_ids) - set(map(str, teacher_ids)))
    missing_x = sorted(set(sample_ids) - set(map(str, x_unnorm.index)))
    missing_meta = sorted(set(sample_ids) - set(metadata["sample_id"].astype(str)))
    if missing_teacher or missing_x or missing_meta:
        raise ValueError(
            "Selected sample IDs missing from sources: "
            f"teacher={missing_teacher[:5]}, x={missing_x[:5]}, metadata={missing_meta[:5]}"
        )

    metadata_by_id = metadata.set_index("sample_id", drop=False)
    selected_metadata = metadata_by_id.loc[sample_ids].reset_index(drop=True)
    selected_x = x_unnorm.loc[sample_ids]
    all_taxa_names = [str(value) for value in selected_x.columns]
    x_values_full = selected_x.to_numpy(dtype=np.float32)
    splits = stratified_split_sample_ids(
        selected_metadata,
        sample_ids=sample_ids,
        label_column=split_label_column,
        seed=seed,
    )
    selected_count = len(all_taxa_names) if num_taxa is None else min(num_taxa, len(all_taxa_names))
    selected_indices, taxa_filter = select_top_taxa_by_train(
        x_values_full,
        splits["split"].tolist(),
        all_taxa_names,
        num_taxa=selected_count,
    )
    x_values, retained_mass = filter_and_normalize_taxa(x_values_full, selected_indices)
    taxa_names = taxa_filter["taxon"].astype(str).tolist()
    keep_mask = retained_mass > 0
    dropped_zero_retained = int((~keep_mask).sum())
    if dropped_zero_retained:
        sample_ids = [sample_id for sample_id, keep in zip(sample_ids, keep_mask) if bool(keep)]
        selected_metadata = selected_metadata.loc[keep_mask].reset_index(drop=True)
        splits = splits.loc[keep_mask].reset_index(drop=True)
        x_values_full = x_values_full[keep_mask]
        x_values = x_values[keep_mask]
        retained_mass = retained_mass[keep_mask]

    pd.DataFrame({"sample_id": sample_ids}).to_csv(output_dir / "sample_ids.csv", index=False)
    selected_metadata.to_csv(output_dir / "metadata.csv", index=False)
    splits.to_csv(output_dir / "sample_splits.csv", index=False)
    pd.DataFrame({"taxon": taxa_names}).to_csv(output_dir / "taxa_names.csv", index=False)
    taxa_filter.to_csv(output_dir / "taxa_filter.csv", index=False)
    np.savez_compressed(
        output_dir / "X_unnorm.npz",
        sample_ids=np.array(sample_ids),
        taxa_names=np.array(all_taxa_names),
        X=x_values_full,
    )
    np.savez_compressed(
        output_dir / "X_taxa.npz",
        sample_ids=np.array(sample_ids),
        taxa_names=np.array(taxa_names),
        source_indices=selected_indices,
        retained_mass=retained_mass,
        X=x_values,
    )

    row_sums = x_values.sum(axis=1)
    manifest = {
        "num_samples": len(sample_ids),
        "num_taxa": len(taxa_names),
        "num_taxa_available": len(all_taxa_names),
        "dropped_zero_retained_samples": dropped_zero_retained,
        "sample_id_first10": sample_ids[:10],
        "taxa_first20": taxa_names[:20],
        "x_shape": list(x_values.shape),
        "x_dtype": str(x_values.dtype),
        "x_row_sum_min": float(row_sums.min()),
        "x_row_sum_max": float(row_sums.max()),
        "x_zero_fraction": float((x_values == 0).sum() / x_values.size),
        "retained_mass_min": float(retained_mass.min()),
        "retained_mass_mean": float(retained_mass.mean()),
        "retained_mass_max": float(retained_mass.max()),
        "split_label_column": split_label_column,
        "split_counts": splits["split"].value_counts().sort_index().to_dict(),
        "alignment": alignment_report(sample_ids, selected_metadata["sample_id"].astype(str).tolist()),
        "outputs": {
            "sample_ids": str(output_dir / "sample_ids.csv"),
            "metadata": str(output_dir / "metadata.csv"),
            "sample_splits": str(output_dir / "sample_splits.csv"),
            "taxa_names": str(output_dir / "taxa_names.csv"),
            "taxa_filter": str(output_dir / "taxa_filter.csv"),
            "X_unnorm": str(output_dir / "X_unnorm.npz"),
            "X_taxa": str(output_dir / "X_taxa.npz"),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def build_pilot_dataset(
    data_root: Path,
    output_dir: Path,
    max_samples: int,
    seed: int,
    label_column: str,
    num_taxa: int | None,
) -> dict[str, object]:
    metadata = normalize_mgnify_metadata(pd.read_csv(data_root / "mgnify_biomes.csv", dtype=str, low_memory=False))

    with (data_root / "MicroCorpus-260K.pkl").open("rb") as handle:
        teacher_corpus = pickle.load(handle)
    teacher_ids = [str(value) for value in teacher_corpus.data.index]
    del teacher_corpus

    with (data_root / "MicroCorpus-260K_unnorm.pkl").open("rb") as handle:
        unnorm_corpus = pickle.load(handle)
    x_unnorm = unnorm_corpus.data

    available_metadata = metadata[metadata["sample_id"].isin(teacher_ids) & metadata["sample_id"].isin(x_unnorm.index)]
    sample_ids = select_balanced_sample_ids(
        available_metadata,
        label_column=label_column,
        max_samples=max_samples,
        seed=seed,
    )
    return write_pilot_dataset(
        sample_ids=sample_ids,
        metadata=metadata,
        teacher_ids=teacher_ids,
        x_unnorm=x_unnorm,
        output_dir=output_dir,
        num_taxa=num_taxa,
        split_label_column=label_column,
        seed=seed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/compass_biome/pilot_dataset"))
    parser.add_argument("--max-samples", type=int, default=1024)
    parser.add_argument("--num-taxa", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label-column", default="biome_1")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_pilot_dataset(
        data_root=args.data_root,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        seed=args.seed,
        label_column=args.label_column,
        num_taxa=args.num_taxa,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
