#!/usr/bin/env python3
"""Train a lightweight ComPASS-Biome bottleneck pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from scripts.compass_biome_utils import (
    biome_enrichment_rows,
    mean_baseline_predictions,
    program_diagnostics_rows,
    reconstruction_metric_rows,
    top_taxa_rows,
)


def _resolve_device(device_name: str):
    import torch

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def usage_balance_loss(activations, mode: str = "kl_uniform_to_usage", eps: float = 1e-8):
    """Penalize uneven mean program usage within a batch."""
    import torch

    if activations.ndim != 2:
        raise ValueError("activations must be a 2D tensor")
    if mode not in {"kl_uniform_to_usage", "mse"}:
        raise ValueError("mode must be one of: kl_uniform_to_usage, mse")
    q = activations.mean(dim=0)
    num_programs = q.shape[0]
    if num_programs <= 0:
        raise ValueError("activations must contain at least one program")
    uniform = torch.full_like(q, 1.0 / num_programs)
    if mode == "mse":
        return ((q - uniform) ** 2).mean()
    return (uniform * (torch.log(uniform.clamp_min(eps)) - torch.log(q.clamp_min(eps)))).sum()


def sample_entropy_target_loss(activations, target_effective_programs: float = 2.0, eps: float = 1e-8):
    """Penalize sample entropy away from a target effective program count."""
    import torch

    if activations.ndim != 2:
        raise ValueError("activations must be a 2D tensor")
    if target_effective_programs <= 0:
        raise ValueError("target_effective_programs must be positive")
    target_entropy = torch.log(torch.tensor(float(target_effective_programs), dtype=activations.dtype, device=activations.device))
    entropy = -(activations * torch.log(activations.clamp_min(eps))).sum(dim=1)
    return ((entropy - target_entropy) ** 2).mean()


def dictionary_diversity_loss(
    taxa_programs,
    mode: str = "hinge_cosine",
    threshold: float = 0.30,
    eps: float = 1e-8,
):
    """Penalize redundant taxa dictionaries by their off-diagonal cosine similarity."""
    import torch

    if taxa_programs.ndim != 2:
        raise ValueError("taxa_programs must be a 2D tensor")
    if mode not in {"hinge_cosine", "mse_offdiag"}:
        raise ValueError("mode must be one of: hinge_cosine, mse_offdiag")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    num_programs = taxa_programs.shape[0]
    if num_programs <= 0:
        raise ValueError("taxa_programs must contain at least one program")
    if num_programs == 1:
        return taxa_programs.sum() * 0.0

    norms = taxa_programs.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
    normalized = taxa_programs / norms
    cosine = normalized @ normalized.T
    offdiag = cosine[~torch.eye(num_programs, dtype=torch.bool, device=taxa_programs.device)]
    if mode == "mse_offdiag":
        return (offdiag ** 2).mean()
    return torch.relu(offdiag - threshold).pow(2).mean()


def dictionary_anchor_loss(anchor_programs, taxa_programs, mode: str = "kl_anchor_to_current", eps: float = 1e-8):
    """Penalize trainable taxa dictionaries for drifting away from an anchor."""
    import torch

    if anchor_programs.ndim != 2 or taxa_programs.ndim != 2:
        raise ValueError("anchor_programs and taxa_programs must be 2D tensors")
    if anchor_programs.shape != taxa_programs.shape:
        raise ValueError("anchor_programs and taxa_programs must have the same shape")
    if mode != "kl_anchor_to_current":
        raise ValueError("mode must be kl_anchor_to_current")
    return (
        anchor_programs.clamp_min(eps)
        * (torch.log(anchor_programs.clamp_min(eps)) - torch.log(taxa_programs.clamp_min(eps)))
    ).sum(dim=1).mean()


def _normalize_taxa_dictionary_init(
    taxa_dictionary_init: np.ndarray,
    num_programs: int,
    num_taxa: int,
) -> np.ndarray:
    init = np.asarray(taxa_dictionary_init, dtype=np.float32)
    if init.shape != (num_programs, num_taxa):
        raise ValueError(
            f"taxa_dictionary_init must have shape {(num_programs, num_taxa)}, got {init.shape}"
        )
    if not np.isfinite(init).all():
        raise ValueError("taxa_dictionary_init must contain finite values")
    if np.any(init < 0):
        raise ValueError("taxa_dictionary_init must be non-negative")
    row_sums = init.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("each taxa_dictionary_init row must have positive mass")
    return (init / row_sums).astype(np.float32)


def train_bottleneck_model(
    embeddings: np.ndarray,
    taxa_targets: np.ndarray,
    num_programs: int = 32,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-2,
    seed: int = 0,
    device_name: str = "auto",
    train_indices: Sequence[int] | None = None,
    validation_indices: Sequence[int] | None = None,
    training_targets: np.ndarray | None = None,
    patience: int | None = None,
    usage_balance_weight: float = 0.0,
    usage_balance_mode: str = "kl_uniform_to_usage",
    sample_entropy_weight: float = 0.0,
    sample_entropy_target_effective: float = 2.0,
    dictionary_diversity_weight: float = 0.0,
    dictionary_diversity_mode: str = "hinge_cosine",
    dictionary_diversity_threshold: float = 0.30,
    taxa_dictionary_init: np.ndarray | None = None,
    freeze_taxa_dictionary: bool = False,
    dictionary_anchor_weight: float = 0.0,
    dictionary_anchor_mode: str = "kl_anchor_to_current",
) -> dict[str, Any]:
    """Train a soft program bottleneck that reconstructs taxa distributions."""
    import torch
    from torch import nn

    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    if taxa_targets.ndim != 2:
        raise ValueError("taxa_targets must be a 2D array")
    if embeddings.shape[0] != taxa_targets.shape[0]:
        raise ValueError("embeddings and taxa_targets must have the same number of rows")
    if num_programs <= 0:
        raise ValueError("num_programs must be positive")
    if training_targets is not None and training_targets.shape != taxa_targets.shape:
        raise ValueError("training_targets must match taxa_targets shape")
    if patience is not None and patience <= 0:
        raise ValueError("patience must be positive when provided")
    if usage_balance_weight < 0:
        raise ValueError("usage_balance_weight must be non-negative")
    if usage_balance_mode not in {"kl_uniform_to_usage", "mse"}:
        raise ValueError("usage_balance_mode must be one of: kl_uniform_to_usage, mse")
    if sample_entropy_weight < 0:
        raise ValueError("sample_entropy_weight must be non-negative")
    if sample_entropy_target_effective <= 0:
        raise ValueError("sample_entropy_target_effective must be positive")
    if sample_entropy_target_effective > num_programs:
        raise ValueError("sample_entropy_target_effective cannot exceed num_programs")
    if dictionary_diversity_weight < 0:
        raise ValueError("dictionary_diversity_weight must be non-negative")
    if dictionary_diversity_mode not in {"hinge_cosine", "mse_offdiag"}:
        raise ValueError("dictionary_diversity_mode must be one of: hinge_cosine, mse_offdiag")
    if dictionary_diversity_threshold < 0:
        raise ValueError("dictionary_diversity_threshold must be non-negative")
    if freeze_taxa_dictionary and taxa_dictionary_init is None:
        raise ValueError("freeze_taxa_dictionary requires taxa_dictionary_init")
    if dictionary_anchor_weight < 0:
        raise ValueError("dictionary_anchor_weight must be non-negative")
    if dictionary_anchor_weight > 0 and taxa_dictionary_init is None:
        raise ValueError("dictionary_anchor_weight requires taxa_dictionary_init")
    if dictionary_anchor_mode != "kl_anchor_to_current":
        raise ValueError("dictionary_anchor_mode must be kl_anchor_to_current")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = _resolve_device(device_name)

    z = torch.tensor(embeddings, dtype=torch.float32, device=device)
    x = torch.tensor(taxa_targets, dtype=torch.float32, device=device)
    x = x / x.sum(dim=1, keepdim=True).clamp_min(1e-12)
    train_target_source = taxa_targets if training_targets is None else training_targets
    train_target = torch.tensor(train_target_source, dtype=torch.float32, device=device)
    train_target = train_target / train_target.sum(dim=1, keepdim=True).clamp_min(1e-12)

    encoder = nn.Linear(z.shape[1], num_programs, bias=True).to(device)
    anchor_programs = None
    if taxa_dictionary_init is None:
        taxa_logits = nn.Parameter(torch.empty(num_programs, x.shape[1], device=device))
        nn.init.normal_(taxa_logits, mean=0.0, std=0.02)
    else:
        normalized_init = _normalize_taxa_dictionary_init(
            taxa_dictionary_init,
            num_programs=num_programs,
            num_taxa=x.shape[1],
        )
        anchor_programs = torch.tensor(normalized_init, dtype=torch.float32, device=device)
        taxa_logits = nn.Parameter(torch.log(anchor_programs.clamp_min(1e-8)), requires_grad=not freeze_taxa_dictionary)
    optimizer_params = list(encoder.parameters())
    if taxa_logits.requires_grad:
        optimizer_params.append(taxa_logits)
    optimizer = torch.optim.Adam(optimizer_params, lr=learning_rate)

    metrics: list[dict[str, float]] = []
    indices = np.arange(z.shape[0]) if train_indices is None else np.asarray(train_indices, dtype=np.int64)
    if len(indices) == 0:
        raise ValueError("at least one training index is required")
    validation_indices_array = (
        None if validation_indices is None else np.asarray(validation_indices, dtype=np.int64)
    )
    best_score = float("inf")
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0
    for epoch in range(epochs):
        rng.shuffle(indices)
        epoch_losses: list[float] = []
        epoch_reconstruction_losses: list[float] = []
        epoch_usage_balance_losses: list[float] = []
        epoch_entropy_losses: list[float] = []
        epoch_dictionary_diversity_losses: list[float] = []
        epoch_dictionary_anchor_losses: list[float] = []
        epoch_sample_entropies: list[float] = []
        for start in range(0, len(indices), batch_size):
            batch_idx = torch.tensor(indices[start : start + batch_size], dtype=torch.long, device=device)
            batch_z = z.index_select(0, batch_idx)
            batch_x = train_target.index_select(0, batch_idx)

            activations = torch.softmax(encoder(batch_z), dim=1)
            taxa_programs = torch.softmax(taxa_logits, dim=1)
            recon = activations @ taxa_programs
            reconstruction_loss = -(batch_x * torch.log(recon.clamp_min(1e-8))).sum(dim=1).mean()
            balance_loss = usage_balance_loss(activations, mode=usage_balance_mode)
            entropy_target_loss = sample_entropy_target_loss(
                activations,
                target_effective_programs=sample_entropy_target_effective,
            )
            diversity_loss = dictionary_diversity_loss(
                taxa_programs,
                mode=dictionary_diversity_mode,
                threshold=dictionary_diversity_threshold,
            )
            anchor_loss = (
                dictionary_anchor_loss(anchor_programs, taxa_programs, mode=dictionary_anchor_mode)
                if anchor_programs is not None
                else taxa_programs.sum() * 0.0
            )
            loss = (
                reconstruction_loss
                + usage_balance_weight * balance_loss
                + sample_entropy_weight * entropy_target_loss
                + dictionary_diversity_weight * diversity_loss
                + dictionary_anchor_weight * anchor_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            epoch_reconstruction_losses.append(float(reconstruction_loss.detach().cpu()))
            epoch_usage_balance_losses.append(float(balance_loss.detach().cpu()))
            epoch_entropy_losses.append(float(entropy_target_loss.detach().cpu()))
            epoch_dictionary_diversity_losses.append(float(diversity_loss.detach().cpu()))
            epoch_dictionary_anchor_losses.append(float(anchor_loss.detach().cpu()))
            batch_entropy = -(activations * torch.log(activations.clamp_min(1e-8))).sum(dim=1).mean()
            epoch_sample_entropies.append(float(batch_entropy.detach().cpu()))

        with torch.no_grad():
            all_a = torch.softmax(encoder(z), dim=1)
            all_p = torch.softmax(taxa_logits, dim=1)
            all_recon = all_a @ all_p
            reconstruction_objective_loss = -(train_target * torch.log(all_recon.clamp_min(1e-8))).sum(dim=1).mean()
            full_balance_loss = usage_balance_loss(all_a, mode=usage_balance_mode)
            full_entropy_target_loss = sample_entropy_target_loss(
                all_a,
                target_effective_programs=sample_entropy_target_effective,
            )
            full_dictionary_diversity_loss = dictionary_diversity_loss(
                all_p,
                mode=dictionary_diversity_mode,
                threshold=dictionary_diversity_threshold,
            )
            full_dictionary_anchor_loss = (
                dictionary_anchor_loss(anchor_programs, all_p, mode=dictionary_anchor_mode)
                if anchor_programs is not None
                else all_p.sum() * 0.0
            )
            full_sample_entropy = -(all_a * torch.log(all_a.clamp_min(1e-8))).sum(dim=1).mean()
            objective_loss = (
                reconstruction_objective_loss
                + usage_balance_weight * full_balance_loss
                + sample_entropy_weight * full_entropy_target_loss
                + dictionary_diversity_weight * full_dictionary_diversity_loss
                + dictionary_anchor_weight * full_dictionary_anchor_loss
            )
            true_loss = -(x * torch.log(all_recon.clamp_min(1e-8))).sum(dim=1).mean()
            train_idx_tensor = torch.tensor(indices, dtype=torch.long, device=device)
            train_loss = -(
                x.index_select(0, train_idx_tensor)
                * torch.log(all_recon.index_select(0, train_idx_tensor).clamp_min(1e-8))
            ).sum(dim=1).mean()
            if validation_indices_array is not None and len(validation_indices_array) > 0:
                valid_idx_tensor = torch.tensor(validation_indices_array, dtype=torch.long, device=device)
                validation_loss = -(
                    x.index_select(0, valid_idx_tensor)
                    * torch.log(all_recon.index_select(0, valid_idx_tensor).clamp_min(1e-8))
                ).sum(dim=1).mean()
                early_stop_score = float(validation_loss.detach().cpu())
            else:
                validation_loss = None
                early_stop_score = float(objective_loss.detach().cpu())
        metrics.append(
            {
                "epoch": float(epoch + 1),
                "loss": float(objective_loss.detach().cpu()),
                "regularized_loss": float(objective_loss.detach().cpu()),
                "reconstruction_loss": float(reconstruction_objective_loss.detach().cpu()),
                "true_loss": float(true_loss.detach().cpu()),
                "train_loss": float(train_loss.detach().cpu()),
                "validation_loss": float(validation_loss.detach().cpu()) if validation_loss is not None else float("nan"),
                "batch_loss_mean": float(np.mean(epoch_losses)),
                "batch_reconstruction_loss_mean": float(np.mean(epoch_reconstruction_losses)),
                "usage_balance_loss": float(full_balance_loss.detach().cpu()),
                "batch_usage_balance_loss_mean": float(np.mean(epoch_usage_balance_losses)),
                "sample_entropy_mean": float(full_sample_entropy.detach().cpu()),
                "sample_entropy_target_loss": float(full_entropy_target_loss.detach().cpu()),
                "batch_sample_entropy_target_loss_mean": float(np.mean(epoch_entropy_losses)),
                "batch_sample_entropy_mean": float(np.mean(epoch_sample_entropies)),
                "dictionary_diversity_loss": float(full_dictionary_diversity_loss.detach().cpu()),
                "batch_dictionary_diversity_loss_mean": float(np.mean(epoch_dictionary_diversity_losses)),
                "dictionary_anchor_loss": float(full_dictionary_anchor_loss.detach().cpu()),
                "batch_dictionary_anchor_loss_mean": float(np.mean(epoch_dictionary_anchor_losses)),
            }
        )
        if early_stop_score < best_score - 1e-7:
            best_score = early_stop_score
            best_state = {
                "encoder": {key: value.detach().cpu().clone() for key, value in encoder.state_dict().items()},
                "taxa_logits": taxa_logits.detach().cpu().clone(),
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if patience is not None and epochs_without_improvement >= patience:
            break

    if best_state is not None:
        encoder.load_state_dict({key: value.to(device) for key, value in best_state["encoder"].items()})
        taxa_logits.data.copy_(best_state["taxa_logits"].to(device))

    with torch.no_grad():
        activations = torch.softmax(encoder(z), dim=1).detach().cpu().numpy().astype(np.float32)
        taxa_programs = torch.softmax(taxa_logits, dim=1).detach().cpu().numpy().astype(np.float32)
        reconstructions = (torch.softmax(encoder(z), dim=1) @ torch.softmax(taxa_logits, dim=1))
        reconstructions = reconstructions.detach().cpu().numpy().astype(np.float32)

    return {
        "activations": activations,
        "taxa_programs": taxa_programs,
        "reconstructions": reconstructions,
        "metrics": metrics,
        "device": str(device),
    }


def _load_npz_array(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return data[key]


def _load_taxa_dictionary_init(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    with np.load(path, allow_pickle=False) as data:
        if "taxa_programs" not in data.files:
            raise ValueError("taxa dictionary init npz must contain taxa_programs")
        return data["taxa_programs"].astype(np.float32)


def _load_taxa_matrix(dataset_dir: Path) -> np.ndarray:
    taxa_path = dataset_dir / "X_taxa.npz"
    if taxa_path.exists():
        return _load_npz_array(taxa_path, "X").astype(np.float32)
    return _load_npz_array(dataset_dir / "X_unnorm.npz", "X").astype(np.float32)


def _load_splits(dataset_dir: Path, sample_ids: Sequence[str], metadata: pd.DataFrame) -> list[str]:
    split_path = dataset_dir / "sample_splits.csv"
    if not split_path.exists():
        return ["train"] * len(sample_ids)
    splits = pd.read_csv(split_path)
    required = {"sample_id", "split"}
    if not required.issubset(splits.columns):
        raise ValueError("sample_splits.csv must contain sample_id and split columns")
    split_ids = splits["sample_id"].astype(str).tolist()
    if split_ids != list(sample_ids):
        raise ValueError("sample_splits.csv sample IDs do not match dataset sample IDs")
    metadata["split"] = splits["split"].astype(str).tolist()
    return splits["split"].astype(str).tolist()


def _shuffle_training_targets(x: np.ndarray, splits: Sequence[str], seed: int) -> np.ndarray:
    shuffled = x.copy()
    train_indices = np.flatnonzero(np.asarray([str(split) for split in splits]) == "train")
    if len(train_indices) > 1:
        rng = np.random.default_rng(seed)
        shuffled[train_indices] = x[rng.permutation(train_indices)]
    return shuffled


def _write_plots(
    output_dir: Path,
    sample_ids: Sequence[str],
    taxa_names: Sequence[str],
    metadata: pd.DataFrame,
    splits: Sequence[str],
    x: np.ndarray,
    reconstructions: np.ndarray,
    activations: np.ndarray,
    taxa_programs: np.ndarray,
    metrics: pd.DataFrame,
    top_k: int,
    label_column: str,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: dict[str, str] = {}

    order = np.lexsort(
        (
            np.asarray(sample_ids, dtype=str),
            np.asarray(metadata[label_column].astype(str) if label_column in metadata.columns else [""] * len(sample_ids)),
            np.asarray([str(split) for split in splits]),
        )
    )
    fig, ax = plt.subplots(figsize=(max(6, activations.shape[1] * 0.45), 6))
    image = ax.imshow(activations[order], aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_xlabel("Program")
    ax.set_ylabel("Samples sorted by split/label")
    ax.set_title("Program usage")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    path = output_dir / "program_usage_heatmap.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["program_usage_heatmap"] = str(path)

    example_indices = np.flatnonzero(np.asarray(splits) == "test")[:4]
    if len(example_indices) == 0:
        example_indices = np.arange(min(4, len(sample_ids)))
    fig, axes = plt.subplots(len(example_indices), 1, figsize=(10, max(3, 2.4 * len(example_indices))))
    if len(example_indices) == 1:
        axes = [axes]
    for ax, row_idx in zip(axes, example_indices):
        true_top = np.argsort(-x[row_idx], kind="mergesort")[:top_k]
        pred_top = np.argsort(-reconstructions[row_idx], kind="mergesort")[:top_k]
        union = list(dict.fromkeys(true_top.tolist() + pred_top.tolist()))[: max(top_k, 1)]
        labels = [str(taxa_names[idx]).replace("g__", "") for idx in union]
        positions = np.arange(len(union))
        ax.bar(positions - 0.18, x[row_idx, union], width=0.36, label="true")
        ax.bar(positions + 0.18, reconstructions[row_idx, union], width=0.36, label="recon")
        ax.set_title(str(sample_ids[row_idx]))
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("abundance")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    path = output_dir / "example_true_vs_reconstructed_taxa.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["example_true_vs_reconstructed_taxa"] = str(path)

    per_program_top = min(8, len(taxa_names))
    fig, axes = plt.subplots(taxa_programs.shape[0], 1, figsize=(9, max(3, 1.4 * taxa_programs.shape[0])))
    if taxa_programs.shape[0] == 1:
        axes = [axes]
    for program_idx, ax in enumerate(axes):
        order_idx = np.argsort(-taxa_programs[program_idx], kind="mergesort")[:per_program_top]
        labels = [str(taxa_names[idx]).replace("g__", "") for idx in order_idx]
        ax.barh(np.arange(len(order_idx)), taxa_programs[program_idx, order_idx])
        ax.set_yticks(np.arange(len(order_idx)))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_title(f"Program {program_idx}")
    fig.tight_layout()
    path = output_dir / "top_taxa_per_program.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["top_taxa_per_program_plot"] = str(path)

    test_metrics = metrics[metrics["split"] == "test"].copy()
    if test_metrics.empty:
        test_metrics = metrics.copy()
    fig, ax = plt.subplots(figsize=(8, 4))
    positions = np.arange(len(test_metrics))
    width = 0.35
    ax.bar(positions - width / 2, test_metrics[f"top_{top_k}_recall"], width=width, label=f"Top-{top_k} recall")
    ax.bar(positions + width / 2, test_metrics["bray_curtis_similarity"], width=width, label="Bray-Curtis sim.")
    ax.set_xticks(positions)
    ax.set_xticklabels(test_metrics["model"].tolist(), rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_title("Reconstruction metrics")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "metrics_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["metrics_comparison"] = str(path)

    return paths


def run_pilot(
    dataset_dir: Path,
    embeddings_path: Path,
    output_dir: Path,
    num_programs: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device_name: str,
    top_k: int,
    label_column: str,
    controls: str = "all",
    patience: int | None = None,
    make_plots: bool = True,
    usage_balance_weight: float = 0.0,
    usage_balance_mode: str = "kl_uniform_to_usage",
    sample_entropy_weight: float = 0.0,
    sample_entropy_target_effective: float = 2.0,
    dictionary_diversity_weight: float = 0.0,
    dictionary_diversity_mode: str = "hinge_cosine",
    dictionary_diversity_threshold: float = 0.30,
    taxa_dictionary_init_path: Path | None = None,
    freeze_taxa_dictionary: bool = False,
    dictionary_anchor_weight: float = 0.0,
    dictionary_anchor_mode: str = "kl_anchor_to_current",
) -> dict[str, object]:
    sample_ids = pd.read_csv(dataset_dir / "sample_ids.csv")["sample_id"].astype(str).tolist()
    taxa_names = pd.read_csv(dataset_dir / "taxa_names.csv")["taxon"].astype(str).tolist()
    metadata = pd.read_csv(dataset_dir / "metadata.csv")
    x = _load_taxa_matrix(dataset_dir)
    splits = _load_splits(dataset_dir, sample_ids, metadata)
    with np.load(embeddings_path, allow_pickle=False) as embedding_data:
        z = embedding_data["embeddings"].astype(np.float32)
        embedding_ids = [str(value) for value in embedding_data["sample_ids"]]

    if embedding_ids != sample_ids:
        raise ValueError("Embedding sample IDs do not match dataset sample IDs")
    if x.shape[0] != z.shape[0]:
        raise ValueError("X and embeddings row counts differ")
    if x.shape[1] != len(taxa_names):
        raise ValueError("X width does not match taxa names")
    if controls not in {"none", "shuffle", "mean", "all"}:
        raise ValueError("controls must be one of: none, shuffle, mean, all")

    train_indices = np.flatnonzero(np.asarray(splits) == "train")
    validation_indices = np.flatnonzero(np.asarray(splits) == "valid")
    if len(train_indices) == 0:
        raise ValueError("sample_splits.csv must contain at least one train sample")
    taxa_dictionary_init = _load_taxa_dictionary_init(taxa_dictionary_init_path)

    result = train_bottleneck_model(
        embeddings=z,
        taxa_targets=x,
        num_programs=num_programs,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        device_name=device_name,
        train_indices=train_indices,
        validation_indices=validation_indices,
        patience=patience,
        usage_balance_weight=usage_balance_weight,
        usage_balance_mode=usage_balance_mode,
        sample_entropy_weight=sample_entropy_weight,
        sample_entropy_target_effective=sample_entropy_target_effective,
        dictionary_diversity_weight=dictionary_diversity_weight,
        dictionary_diversity_mode=dictionary_diversity_mode,
        dictionary_diversity_threshold=dictionary_diversity_threshold,
        taxa_dictionary_init=taxa_dictionary_init,
        freeze_taxa_dictionary=freeze_taxa_dictionary,
        dictionary_anchor_weight=dictionary_anchor_weight,
        dictionary_anchor_mode=dictionary_anchor_mode,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "bottleneck_outputs.npz",
        sample_ids=np.array(sample_ids),
        activations=result["activations"],
        taxa_programs=result["taxa_programs"],
        reconstructions=result["reconstructions"],
        splits=np.array(splits),
    )
    pd.DataFrame(result["metrics"]).to_csv(output_dir / "training_metrics.csv", index=False)
    pd.DataFrame(result["metrics"]).to_csv(output_dir / "metrics.csv", index=False)

    metric_rows = reconstruction_metric_rows(
        x,
        result["reconstructions"],
        splits,
        model_name="real",
        top_k=top_k,
    )
    control_outputs: dict[str, str] = {}
    if controls in {"mean", "all"}:
        mean_predictions = mean_baseline_predictions(x, splits)
        metric_rows.extend(
            reconstruction_metric_rows(
                x,
                mean_predictions,
                splits,
                model_name="mean_baseline",
                top_k=top_k,
            )
        )
        np.savez_compressed(
            output_dir / "mean_baseline_outputs.npz",
            sample_ids=np.array(sample_ids),
            reconstructions=mean_predictions,
            splits=np.array(splits),
        )
        control_outputs["mean_baseline_outputs"] = str(output_dir / "mean_baseline_outputs.npz")
    if controls in {"shuffle", "all"}:
        shuffled_targets = _shuffle_training_targets(x, splits, seed=seed + 17)
        shuffle_result = train_bottleneck_model(
            embeddings=z,
            taxa_targets=x,
            training_targets=shuffled_targets,
            num_programs=num_programs,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed + 1,
            device_name=device_name,
            train_indices=train_indices,
            validation_indices=validation_indices,
            patience=patience,
            usage_balance_weight=usage_balance_weight,
            usage_balance_mode=usage_balance_mode,
            sample_entropy_weight=sample_entropy_weight,
            sample_entropy_target_effective=sample_entropy_target_effective,
            dictionary_diversity_weight=dictionary_diversity_weight,
            dictionary_diversity_mode=dictionary_diversity_mode,
            dictionary_diversity_threshold=dictionary_diversity_threshold,
            taxa_dictionary_init=taxa_dictionary_init,
            freeze_taxa_dictionary=freeze_taxa_dictionary,
            dictionary_anchor_weight=dictionary_anchor_weight,
            dictionary_anchor_mode=dictionary_anchor_mode,
        )
        metric_rows.extend(
            reconstruction_metric_rows(
                x,
                shuffle_result["reconstructions"],
                splits,
                model_name="shuffle",
                top_k=top_k,
            )
        )
        np.savez_compressed(
            output_dir / "shuffle_bottleneck_outputs.npz",
            sample_ids=np.array(sample_ids),
            activations=shuffle_result["activations"],
            taxa_programs=shuffle_result["taxa_programs"],
            reconstructions=shuffle_result["reconstructions"],
            splits=np.array(splits),
        )
        pd.DataFrame(shuffle_result["metrics"]).to_csv(output_dir / "shuffle_training_metrics.csv", index=False)
        control_outputs["shuffle_bottleneck_outputs"] = str(output_dir / "shuffle_bottleneck_outputs.npz")
        control_outputs["shuffle_training_metrics"] = str(output_dir / "shuffle_training_metrics.csv")

    reconstruction_metrics = pd.DataFrame(metric_rows)
    reconstruction_metrics.to_csv(output_dir / "reconstruction_metrics.csv", index=False)

    pd.DataFrame(program_diagnostics_rows(result["activations"], result["taxa_programs"])).to_csv(
        output_dir / "program_diagnostics.csv", index=False
    )
    top_taxa = pd.DataFrame(top_taxa_rows(result["taxa_programs"], taxa_names, top_k))
    top_taxa.to_csv(output_dir / "top_taxa_per_program.csv", index=False)
    top_taxa.to_csv(output_dir / "top_taxa.csv", index=False)
    if label_column in metadata.columns:
        pd.DataFrame(biome_enrichment_rows(result["activations"], metadata, label_column)).to_csv(
            output_dir / "biome_enrichment.csv", index=False
        )
    plot_outputs: dict[str, str] = {}
    if make_plots:
        plot_outputs = _write_plots(
            output_dir=output_dir,
            sample_ids=sample_ids,
            taxa_names=taxa_names,
            metadata=metadata,
            splits=splits,
            x=x,
            reconstructions=result["reconstructions"],
            activations=result["activations"],
            taxa_programs=result["taxa_programs"],
            metrics=reconstruction_metrics,
            top_k=top_k,
            label_column=label_column,
        )

    manifest = {
        "dataset_dir": str(dataset_dir),
        "embeddings_path": str(embeddings_path),
        "num_samples": len(sample_ids),
        "embedding_width": int(z.shape[1]),
        "num_taxa": int(x.shape[1]),
        "num_programs": num_programs,
        "epochs": epochs,
        "epochs_ran": len(result["metrics"]),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "device": result["device"],
        "controls": controls,
        "patience": patience,
        "usage_balance_weight": usage_balance_weight,
        "usage_balance_loss": usage_balance_mode,
        "sample_entropy_weight": sample_entropy_weight,
        "sample_entropy_target_effective": sample_entropy_target_effective,
        "dictionary_diversity_weight": dictionary_diversity_weight,
        "dictionary_diversity_loss_type": dictionary_diversity_mode,
        "dictionary_diversity_threshold": dictionary_diversity_threshold,
        "taxa_dictionary_init": str(taxa_dictionary_init_path) if taxa_dictionary_init_path is not None else None,
        "freeze_taxa_dictionary": freeze_taxa_dictionary,
        "dictionary_anchor_weight": dictionary_anchor_weight,
        "dictionary_anchor_loss": dictionary_anchor_mode,
        "final_loss": result["metrics"][-1]["loss"],
        "split_counts": pd.Series(splits).value_counts().sort_index().to_dict(),
        "outputs": {
            "bottleneck_outputs": str(output_dir / "bottleneck_outputs.npz"),
            "training_metrics": str(output_dir / "training_metrics.csv"),
            "metrics": str(output_dir / "metrics.csv"),
            "reconstruction_metrics": str(output_dir / "reconstruction_metrics.csv"),
            "program_diagnostics": str(output_dir / "program_diagnostics.csv"),
            "top_taxa_per_program": str(output_dir / "top_taxa_per_program.csv"),
            "top_taxa": str(output_dir / "top_taxa.csv"),
            "biome_enrichment": str(output_dir / "biome_enrichment.csv"),
            **control_outputs,
            **plot_outputs,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/compass_biome/bottleneck_pilot"))
    parser.add_argument("--num-programs", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--label-column", default="biome_1")
    parser.add_argument("--controls", choices=["none", "shuffle", "mean", "all"], default="all")
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--usage-balance-weight", type=float, default=0.0)
    parser.add_argument(
        "--usage-balance-loss",
        choices=["kl_uniform_to_usage", "mse"],
        default="kl_uniform_to_usage",
    )
    parser.add_argument("--sample-entropy-weight", type=float, default=0.0)
    parser.add_argument("--sample-entropy-target-effective", type=float, default=2.0)
    parser.add_argument("--dictionary-diversity-weight", type=float, default=0.0)
    parser.add_argument(
        "--dictionary-diversity-loss",
        choices=["hinge_cosine", "mse_offdiag"],
        default="hinge_cosine",
    )
    parser.add_argument("--dictionary-diversity-threshold", type=float, default=0.30)
    parser.add_argument("--taxa-dictionary-init", type=Path, default=None)
    parser.add_argument("--freeze-taxa-dictionary", action="store_true")
    parser.add_argument("--dictionary-anchor-weight", type=float, default=0.0)
    parser.add_argument(
        "--dictionary-anchor-loss",
        choices=["kl_anchor_to_current"],
        default="kl_anchor_to_current",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_pilot(
        dataset_dir=args.dataset_dir,
        embeddings_path=args.embeddings,
        output_dir=args.output_dir,
        num_programs=args.num_programs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device_name=args.device,
        top_k=args.top_k,
        label_column=args.label_column,
        controls=args.controls,
        patience=args.patience,
        make_plots=not args.no_plots,
        usage_balance_weight=args.usage_balance_weight,
        usage_balance_mode=args.usage_balance_loss,
        sample_entropy_weight=args.sample_entropy_weight,
        sample_entropy_target_effective=args.sample_entropy_target_effective,
        dictionary_diversity_weight=args.dictionary_diversity_weight,
        dictionary_diversity_mode=args.dictionary_diversity_loss,
        dictionary_diversity_threshold=args.dictionary_diversity_threshold,
        taxa_dictionary_init_path=args.taxa_dictionary_init,
        freeze_taxa_dictionary=args.freeze_taxa_dictionary,
        dictionary_anchor_weight=args.dictionary_anchor_weight,
        dictionary_anchor_mode=args.dictionary_anchor_loss,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
