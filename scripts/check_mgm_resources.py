#!/usr/bin/env python3
"""Validate MGM packaged resources and run a tiny model forward pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_RESOURCE_FILES = [
    "MicroTokenizer.pkl",
    "general_model/config.json",
    "general_model/generation_config.json",
    "general_model/pytorch_model.bin",
]


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def verify_required_files(repo_root: Path) -> dict[str, dict[str, object]]:
    resources_root = repo_root / "mgm" / "resources"
    report: dict[str, dict[str, object]] = {}
    for rel_path in REQUIRED_RESOURCE_FILES:
        path = resources_root / rel_path
        report[rel_path] = {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
        }
    return report


def _all_required_files_exist(file_report: dict[str, dict[str, object]]) -> bool:
    return all(bool(item["exists"]) and int(item["size_bytes"]) > 0 for item in file_report.values())


def run_model_forward_check(repo_root: Path, device_name: str = "auto") -> dict[str, Any]:
    import torch
    import transformers
    from transformers import GPT2LMHeadModel

    from mgm.src.utils import CustomUnpickler

    resources_root = repo_root / "mgm" / "resources"
    tokenizer_path = resources_root / "MicroTokenizer.pkl"
    model_path = resources_root / "general_model"

    with tokenizer_path.open("rb") as handle:
        tokenizer = CustomUnpickler(handle).load()

    model = GPT2LMHeadModel.from_pretrained(model_path)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    model = model.to(device)
    model.eval()

    input_ids = torch.tensor(
        [[tokenizer.bos_token_id, tokenizer.eos_token_id] + [tokenizer.pad_token_id] * 510],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    attention_mask[:, :2] = 1

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
        )

    return {
        "python_device": str(device),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "tokenizer_vocab_size": int(tokenizer.vocab_size),
        "model_vocab_size": int(model.config.vocab_size),
        "logits_shape": list(outputs.logits.shape),
        "logits_all_finite": bool(torch.isfinite(outputs.logits).all().item()),
        "num_hidden_states": len(outputs.hidden_states),
        "last_hidden_state_shape": list(outputs.hidden_states[-1].shape),
        "last_hidden_state_all_finite": bool(torch.isfinite(outputs.hidden_states[-1]).all().item()),
        "num_attentions": len(outputs.attentions),
        "first_attention_shape": list(outputs.attentions[0].shape),
        "first_attention_all_finite": bool(torch.isfinite(outputs.attentions[0]).all().item()),
    }


def build_report(repo_root: Path, device_name: str = "auto", skip_forward: bool = False) -> dict[str, Any]:
    resource_files = verify_required_files(repo_root)
    report: dict[str, Any] = {
        "repo_root": str(repo_root),
        "resource_files": resource_files,
        "required_files_ok": _all_required_files_exist(resource_files),
    }
    if not skip_forward:
        report["forward_check"] = run_model_forward_check(repo_root, device_name)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_repo_root())
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-forward", action="store_true")
    parser.add_argument("--json-output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(args.repo_root, args.device, args.skip_forward)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(payload)
    print(payload)
    return 0 if report["required_files_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
