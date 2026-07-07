#!/usr/bin/env python3
"""Extract frozen MGM teacher embeddings for a ComPASS-Biome pilot dataset."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from scripts.audit_microcorpus_260k import DEFAULT_DATA_ROOT


def resolve_sample_positions(sample_ids: Sequence[str], corpus_index: pd.Index) -> list[int]:
    """Resolve sample IDs to corpus row positions, preserving requested order."""
    positions = corpus_index.get_indexer([str(sample_id) for sample_id in sample_ids])
    missing = [sample_id for sample_id, pos in zip(sample_ids, positions) if pos < 0]
    if missing:
        raise ValueError(f"Sample IDs missing from corpus index: {missing[:10]}")
    return [int(pos) for pos in positions]


def last_valid_token_embeddings(last_hidden, attention_mask):
    """Select the hidden state at each row's final non-pad token."""
    import torch

    lengths = attention_mask.long().sum(dim=1).clamp_min(1)
    last_positions = lengths - 1
    batch_positions = torch.arange(last_hidden.shape[0], device=last_hidden.device)
    return last_hidden[batch_positions, last_positions]


def _resolve_device(device_name: str):
    import torch

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def extract_embeddings(
    corpus_path: Path,
    sample_ids_path: Path,
    model_path: Path,
    output_dir: Path,
    batch_size: int = 16,
    device_name: str = "auto",
    max_samples: int | None = None,
) -> dict[str, object]:
    import torch
    from transformers import AutoModel

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    sample_ids = pd.read_csv(sample_ids_path)["sample_id"].astype(str).tolist()
    if max_samples is not None:
        sample_ids = sample_ids[:max_samples]

    with corpus_path.open("rb") as handle:
        corpus = pickle.load(handle)
    positions = resolve_sample_positions(sample_ids, corpus.data.index)

    device = _resolve_device(device_name)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()

    embeddings: list[np.ndarray] = []
    pad_token_id = int(corpus.tokenizer.pad_token_id)
    with torch.no_grad():
        for start in range(0, len(positions), batch_size):
            batch_positions = positions[start : start + batch_size]
            input_ids = corpus.tokens[batch_positions].to(device)
            attention_mask = (input_ids != pad_token_id).to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            batch_embeddings = last_valid_token_embeddings(outputs.hidden_states[-1], attention_mask)
            embeddings.append(batch_embeddings.detach().cpu().numpy().astype(np.float32))

    embedding_matrix = np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, 0), dtype=np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "embeddings.npz"
    np.savez_compressed(
        embeddings_path,
        sample_ids=np.array(sample_ids),
        embeddings=embedding_matrix,
    )
    pd.DataFrame(embedding_matrix, columns=[f"emb_{idx}" for idx in range(embedding_matrix.shape[1])]).assign(
        sample_id=sample_ids
    )[["sample_id"] + [f"emb_{idx}" for idx in range(embedding_matrix.shape[1])]].to_csv(
        output_dir / "embeddings.csv", index=False
    )

    finite = bool(np.isfinite(embedding_matrix).all())
    manifest = {
        "corpus_path": str(corpus_path),
        "sample_ids_path": str(sample_ids_path),
        "model_path": str(model_path),
        "device": str(device),
        "batch_size": batch_size,
        "num_samples": len(sample_ids),
        "embedding_width": int(embedding_matrix.shape[1]) if embedding_matrix.ndim == 2 else 0,
        "all_finite": finite,
        "outputs": {
            "embeddings_npz": str(embeddings_path),
            "embeddings_csv": str(output_dir / "embeddings.csv"),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_DATA_ROOT / "MicroCorpus-260K.pkl")
    parser.add_argument("--sample-ids", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("mgm/resources/general_model"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/compass_biome/embeddings"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = extract_embeddings(
        corpus_path=args.corpus,
        sample_ids_path=args.sample_ids,
        model_path=args.model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device_name=args.device,
        max_samples=args.max_samples,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
