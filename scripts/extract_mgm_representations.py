#!/usr/bin/env python3
"""Extract MGM sample embeddings and genus attention top-k rows."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any, Iterable


SPECIAL_TOKENS = {"<pad>", "<mask>", "<bos>", "<eos>"}


def topk_attention_rows(
    sample_id: str,
    token_names: Iterable[str],
    scores: Iterable[float],
    k: int,
) -> list[dict[str, object]]:
    pairs = [
        (token, float(score))
        for token, score in zip(token_names, scores)
        if token not in SPECIAL_TOKENS
    ]
    pairs.sort(key=lambda item: item[1], reverse=True)
    return [
        {"sample_id": sample_id, "rank": rank, "token": token, "attention": score}
        for rank, (token, score) in enumerate(pairs[:k], start=1)
    ]


def _load_corpus(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _resolve_device(device_name: str):
    import torch

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _token_names_for_ids(tokenizer: Any, token_ids: list[int]) -> list[str]:
    return [tokenizer._convert_id_to_token(int(token_id)) for token_id in token_ids]


def _normalize_positive_scores(scores: list[float]) -> list[float]:
    total = sum(score for score in scores if score > 0)
    if total <= 0:
        return scores
    return [score / total if score > 0 else 0.0 for score in scores]


def extract_representations(
    corpus_path: Path,
    model_path: Path,
    output_dir: Path,
    max_samples: int | None = None,
    top_k: int = 10,
    device_name: str = "auto",
) -> dict[str, object]:
    import torch
    from transformers import AutoModel

    corpus = _load_corpus(corpus_path)
    sample_ids = list(corpus.data.index)
    if max_samples is not None:
        sample_ids = sample_ids[:max_samples]

    device = _resolve_device(device_name)
    model = AutoModel.from_pretrained(model_path)
    model = model.to(device)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "sample_embeddings.csv"
    attention_path = output_dir / "attention_topk.csv"

    embedding_rows: list[list[object]] = []
    attention_rows: list[dict[str, object]] = []
    embedding_width: int | None = None

    with torch.no_grad():
        for idx, sample_id in enumerate(sample_ids):
            item = corpus[idx]
            input_ids = item["input_ids"].unsqueeze(0).to(device)
            attention_mask = item["attention_mask"].unsqueeze(0).to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
            )

            valid_positions = attention_mask.squeeze(0).bool()
            last_hidden = outputs.hidden_states[-1].squeeze(0)
            sample_embedding = last_hidden[valid_positions][-1].detach().cpu().tolist()
            embedding_width = len(sample_embedding)
            embedding_rows.append([sample_id] + [float(value) for value in sample_embedding])

            valid_token_ids = input_ids.squeeze(0)[valid_positions].detach().cpu().tolist()
            token_names = _token_names_for_ids(corpus.tokenizer, valid_token_ids)
            attention = sum(outputs.attentions)
            token_scores = attention.sum(dim=1).sum(dim=1).squeeze(0)[valid_positions]
            score_values = _normalize_positive_scores(token_scores.detach().cpu().tolist())
            attention_rows.extend(topk_attention_rows(sample_id, token_names, score_values, top_k))

    with embeddings_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["SampleID"] + [f"emb_{i}" for i in range(embedding_width or 0)])
        writer.writerows(embedding_rows)

    with attention_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "rank", "token", "attention"])
        writer.writeheader()
        writer.writerows(attention_rows)

    manifest = {
        "corpus_path": str(corpus_path),
        "model_path": str(model_path),
        "device": str(device),
        "num_samples": len(sample_ids),
        "embedding_width": embedding_width,
        "top_k": top_k,
        "outputs": {
            "sample_embeddings": str(embeddings_path),
            "attention_topk": str(attention_path),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("mgm/resources/general_model"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/mgm_repro/representations"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = extract_representations(
        corpus_path=args.corpus,
        model_path=args.model,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        top_k=args.top_k,
        device_name=args.device,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
