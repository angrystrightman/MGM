# ComPASS-Biome Experiment Ledger

## Purpose

This file is the curated, continuously updated result ledger for ComPASS-Biome
validation experiments. It summarizes experiment goals, expected outcomes,
observed results, numeric evidence, source artifact links, and current
interpretation.

Full source tables, manifests, plots, and large generated outputs remain under
`runs/compass_biome/`. This ledger intentionally embeds only compact numeric
tables needed for interpretation.

## Current State

- Dataset: full MicroCorpus-260K usable subset, `N=263065`, `G=1000`.
- Teacher: frozen MGM sample embeddings, width `256`.
- Task: taxa abundance readout sanity check from frozen MGM embeddings.
- Current primary model: `K32_balance005_entropy2_div002_thr025`.
- Current objective: taxa reconstruction plus usage balance, local entropy
  target, and weak dictionary diversity.

## Current Best Judgment

- Primary setting: `K=32 + usage balance + weak dictionary diversity`.
- Compact setting: `K=16` under the regularized K sweep.
- `K=64` has the strongest raw reconstruction but is not preferred because
  dead-program fraction and dictionary redundancy worsen.
- `K32_nmf_fixed` is the cleanest NMF-signature probe of MGM embeddings, but it
  is not the best Top-20 taxa recovery model.
- The architecture-level sanity check is positive: MGM embeddings carry
  recoverable taxa-composition signal and the bottleneck can expose readable
  program dictionaries.
- Biological mechanism, disease, pathway, and cross-cohort claims remain out
  of scope for the current MicroCorpus-260K validation.

## Experiment Index

| Experiment | Main run(s) | Source artifact | Current judgment |
|---|---|---|---|
| Full260K taxa readout baseline | `taxa_readout_validation_full260k/bottleneck` | [reconstruction metrics](../runs/compass_biome/taxa_readout_validation_full260k/bottleneck/reconstruction_metrics.csv), [diagnostics](../runs/compass_biome/taxa_readout_validation_full260k/bottleneck/program_diagnostics.csv) | Positive reconstruction signal, severe unregularized collapse. |
| Usage balancing | `K32_baseline`, `K8_compact`, `K32_balance005_entropy2` | [usage balance summary](../runs/compass_biome/taxa_readout_validation_full260k_usage_balance_summary.csv) | Balancing improves reconstruction and collapse diagnostics. |
| Dictionary diversity | `K32_div005_thr030`, `K32_balance005_entropy2_div002_thr025` | [dictionary diversity summary](../runs/compass_biome/taxa_readout_validation_full260k_dictionary_diversity_summary.csv) | Diversity alone helps redundancy; balanced + weak diversity is best overall. |
| Regularized K sweep | `K4`, `K8`, `K16`, `K32`, `K64` | [K sweep summary](../runs/compass_biome/taxa_readout_validation_full260k_regularized_k_sweep_summary.csv), [recommendation](../runs/compass_biome/taxa_readout_validation_full260k_regularized_k_recommendation.json) | Primary `K=32`; compact `K=16`; do not prefer `K=64`. |
| NMF warm start | `K32_nmf_fixed`, `K32_nmf_trainable`, `K32_nmf_anchor005` | [NMF summary](../runs/compass_biome/taxa_readout_validation_full260k_K32_nmf_warm_start_summary.csv), [recommendation](../runs/compass_biome/taxa_readout_validation_full260k_K32_nmf_warm_start_recommendation.json) | Useful tradeoff and NMF probe; not a strict replacement for parent K32. |

## Experiment 1: Full260K Taxa Readout Baseline

Goal: test whether frozen MGM embeddings preserve enough taxa-composition
information to reconstruct fixed-order Top1000 taxa abundance.

Expected result: the real `Z -> A -> X_hat` reconstruction should beat both
train-mean and shuffled `Z-X` controls on held-out test samples.

Source artifacts:
[reconstruction metrics](../runs/compass_biome/taxa_readout_validation_full260k/bottleneck/reconstruction_metrics.csv),
[program diagnostics](../runs/compass_biome/taxa_readout_validation_full260k/bottleneck/program_diagnostics.csv).

| Test model | Top-20 recall | Bray-Curtis |
|---|---:|---:|
| real | 0.4543 | 0.3306 |
| mean baseline | 0.2187 | 0.1249 |
| shuffle | 0.2189 | 0.1248 |

| Diagnostic | Value |
|---|---:|
| dead program fraction | 0.7500 |
| effective programs mean | 1.4863 |
| activation entropy mean | 0.3247 |
| dictionary cosine mean offdiag | 0.5176 |

Judgment: the reconstruction signal is clearly positive because real
held-out metrics beat both controls. The unregularized K32 bottleneck is not
interpretable enough because most programs are dead and dictionaries are
redundant.

## Experiment 2: Usage Balancing

Goal: reduce program collapse while preserving taxa reconstruction signal.

Expected result: global usage balance and local entropy targeting should reduce
dead programs, increase effective program usage, and keep real reconstruction
well above shuffle control.

Source artifact:
[usage balance summary](../runs/compass_biome/taxa_readout_validation_full260k_usage_balance_summary.csv).

| Run | Real Bray-Curtis | Shuffle Bray-Curtis | Dead fraction | Effective programs |
|---|---:|---:|---:|---:|
| `K32_baseline` | 0.3306 | 0.1248 | 0.7500 | 1.4863 |
| `K8_compact` | 0.2705 | 0.1250 | 0.5000 | 1.2235 |
| `K32_balance005_entropy2` | 0.4414 | 0.1266 | 0.1563 | 2.1754 |

Judgment: usage balancing is strongly positive. It improves reconstruction
and substantially reduces collapse without approaching shuffle-control
behavior.

## Experiment 3: Dictionary Diversity

Goal: reduce redundant taxa dictionaries so different programs learn more
distinct taxa profiles.

Expected result: hinge-cosine dictionary diversity should reduce dictionary
cosine and top-taxa overlap. The best setting should preserve the real-control
reconstruction gap while improving redundancy diagnostics.

Source artifact:
[dictionary diversity summary](../runs/compass_biome/taxa_readout_validation_full260k_dictionary_diversity_summary.csv).

| Run | Top-20 recall | Bray-Curtis | Dead fraction | Dict p95 cosine | Top20 overlap mean |
|---|---:|---:|---:|---:|---:|
| `K32_baseline` | 0.4543 | 0.3306 | 0.7500 | 0.9909 | 1.9677 |
| `K32_div005_thr030` | 0.4560 | 0.3278 | 0.7500 | 0.2896 | 1.4335 |
| `K32_balance005_entropy2` | 0.5422 | 0.4414 | 0.1563 | 0.1591 | 0.9597 |
| `K32_balance005_entropy2_div002_thr025` | 0.5440 | 0.4437 | 0.1250 | 0.1537 | 0.9617 |

Judgment: diversity-only regularization reduces dictionary redundancy but does
not solve usage collapse. Balanced plus weak diversity is the best current
overall setting because it combines strong reconstruction, improved usage, and
low redundancy.

## Experiment 4: Regularized K Sweep

Goal: choose a practical program resolution under the current best regularized
objective.

Expected result: smaller K should be more compact but less reconstructive;
larger K should improve reconstruction but risk dead programs or redundant
dictionaries.

Source artifacts:
[K sweep summary](../runs/compass_biome/taxa_readout_validation_full260k_regularized_k_sweep_summary.csv),
[K recommendation](../runs/compass_biome/taxa_readout_validation_full260k_regularized_k_recommendation.json).

| Run | Top-20 recall | Bray-Curtis | Dead fraction | Effective programs | Dict p95 cosine |
|---|---:|---:|---:|---:|---:|
| `K4` | 0.3990 | 0.2712 | 0.0000 | 1.3046 | 0.0635 |
| `K8` | 0.4549 | 0.3179 | 0.0000 | 1.5218 | 0.1240 |
| `K16` | 0.4884 | 0.3786 | 0.0000 | 1.9569 | 0.1415 |
| `K32` | 0.5440 | 0.4437 | 0.1250 | 2.2293 | 0.1537 |
| `K64` | 0.5783 | 0.4933 | 0.4531 | 2.4595 | 0.2396 |

Recommendation summary from JSON:

| Field | Value |
|---|---|
| primary K | `32` |
| compact K | `16` |
| prefer K64 over K32 | `false` |
| primary reason | Smallest K meeting reconstruction, control-gap, usage, and dictionary gates. |

Judgment: `K=32` is the best practical resolution. `K=16` is the compact
setting when fewer programs are more important than peak reconstruction. `K=64`
is not preferred despite better reconstruction because usage and dictionary
diagnostics worsen.

## Experiment 5: NMF Warm Start

Goal: test whether train-split-only NMF taxa signatures are useful as a
dictionary prior or oracle-style interpretability probe for the current primary
K32 setting.

Expected result: NMF warm start should improve dictionary readability and may
improve reconstruction. A strict positive would beat the parent K32 on both
Top-20 recall and Bray-Curtis while preserving usage and dictionary gates.

Source artifacts:
[NMF warm-start summary](../runs/compass_biome/taxa_readout_validation_full260k_K32_nmf_warm_start_summary.csv),
[NMF recommendation](../runs/compass_biome/taxa_readout_validation_full260k_K32_nmf_warm_start_recommendation.json).

| Run | Mode | Top-20 recall | Bray-Curtis | Dead fraction | Dict p95 cosine | NMF drift |
|---|---|---:|---:|---:|---:|---:|
| `K32_parent` | parent | 0.5440 | 0.4437 | 0.1250 | 0.1537 | NA |
| `NMF_oracle` | oracle | 0.5357 | 0.5266 | NA | 0.0208 | 0.0000 |
| `K32_nmf_fixed` | fixed | 0.5268 | 0.4603 | 0.1250 | 0.0208 | 0.0000 |
| `K32_nmf_trainable` | trainable | 0.5342 | 0.4596 | 0.0938 | 0.0901 | 0.2275 |
| `K32_nmf_anchor005` | anchor | 0.5342 | 0.4596 | 0.1250 | 0.0712 | 0.1342 |

Recommendation summary from JSON:

| Field | Value |
|---|---|
| verdict | `tradeoff` |
| recommended run | `K32_nmf_fixed` |
| recommended mode | `nmf_fixed` |
| fixed interpretability positive | `true` |
| reason | No NMF mode beat the parent on both Top-20 recall and Bray-Curtis; best mode improves one axis. |

Judgment: NMF warm start is a useful tradeoff, not a strict replacement.
It improves Bray-Curtis and dictionary interpretability, while the parent K32
balanced plus weak-diversity model remains best for Top-20 recovery. The fixed
NMF readout is the cleanest probe of whether MGM embeddings can predict
traditional NMF-like community signatures.

## Open Questions / Next Experiments

- Inspect top taxa per program for the current primary K32 model.
- Inspect biome-enrichment stability and whether active programs map to
  coherent MGnify source labels.
- Compare program stability across random seeds.
- Decide whether compact K16 should be used for easier manual interpretation.
- Add curated disease, pathway, and cohort datasets later; MicroCorpus-260K
  does not support those claims directly.

## Evidence Artifacts

- [Full260K baseline reconstruction metrics](../runs/compass_biome/taxa_readout_validation_full260k/bottleneck/reconstruction_metrics.csv)
- [Full260K baseline program diagnostics](../runs/compass_biome/taxa_readout_validation_full260k/bottleneck/program_diagnostics.csv)
- [Usage balance summary](../runs/compass_biome/taxa_readout_validation_full260k_usage_balance_summary.csv)
- [Dictionary diversity summary](../runs/compass_biome/taxa_readout_validation_full260k_dictionary_diversity_summary.csv)
- [Regularized K sweep summary](../runs/compass_biome/taxa_readout_validation_full260k_regularized_k_sweep_summary.csv)
- [Regularized K sweep recommendation](../runs/compass_biome/taxa_readout_validation_full260k_regularized_k_recommendation.json)
- [NMF warm-start summary](../runs/compass_biome/taxa_readout_validation_full260k_K32_nmf_warm_start_summary.csv)
- [NMF warm-start recommendation](../runs/compass_biome/taxa_readout_validation_full260k_K32_nmf_warm_start_recommendation.json)
