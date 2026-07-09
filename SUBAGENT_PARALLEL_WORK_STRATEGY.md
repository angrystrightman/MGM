# Subagent / Parallel Work Strategy

## Default Execution

Use a single coordinating agent for implementation and verification. MGM's
environment, smoke data, training outputs, and representation outputs are
coupled enough that uncoordinated parallel edits can create confusing state.
GPU smoke runs are also serialized because they clean and recreate the shared
`runs/mgm_repro/infant_smoke` directory by default.

## Safe Parallel Work

Parallel workers are safe for read-only or output-isolated tasks:

- Inspecting README, notebook cells, package metadata, and CLI source.
- Checking GPU availability and conda environment status.
- Reviewing generated logs after a run completes.
- Drafting documentation sections that do not modify the same files.
- Inspecting MicroCorpus-260K metadata, as long as workers do not mutate or
  launch full pretraining jobs.
- Reviewing ComPASS-Biome audit reports, pilot manifests, and bottleneck
  summary tables after commands complete.
- Reviewing taxa readout CSV/PNG reports after a validation run completes.
- Comparing real, shuffle, and mean-baseline metrics from immutable generated
  outputs.

## Avoid Parallel Writes

Do not run parallel workers that modify:

- `scripts/run_infant_smoke.sh`
- `scripts/prepare_infant_smoke_data.py`
- `scripts/check_mgm_resources.py`
- `scripts/extract_mgm_representations.py`
- `PLAN.md`
- `implementation-notes.md`

These files define the workflow contract and should be changed through one
reviewed sequence.

Do not parallelize commands that write under `runs/mgm_repro/infant_smoke` or
touch `/home/sunyirong/shared/sunyirong/MicroCorpus-260K`.

For ComPASS-Biome, do not parallelize commands that write under the same
`runs/compass_biome/<stage>` directory. Embedding extraction and bottleneck
training should remain serialized because they share GPU and output state.

For the taxa readout sanity check, keep these stages serialized:

- pilot dataset construction, because it defines sample IDs, splits, taxa order,
  and retained-mass filtering for downstream artifacts.
- frozen MGM embedding extraction, because it uses GPU and writes the shared
  embedding cache for a run.
- bottleneck training/evaluation, because real and shuffle controls share output
  naming, GPU state, and metrics manifests.

For full260K K-sweep reruns, reuse the existing dataset and embeddings, but run
each bottleneck command serially on the GPU. Different K values may write to
separate output directories, but concurrent runs would compete for GPU memory
and make resource failures harder to interpret. Read-only comparison of
completed K-sweep CSV/PNG outputs is safe to parallelize.

For usage-balancing or other regularized full260K bottleneck experiments, keep
the real/shuffle training command serialized for the same reason. Parallel work
is safe only after the run completes, for read-only comparison of manifests,
metrics, diagnostics, and plots across baseline, compact, and regularized
outputs.

For dictionary-diversity bottleneck experiments, keep each full260K run
serialized even when output directories differ. The diversity-only and
balanced-plus-diversity commands both train real and shuffle models on the same
GPU-scale artifacts, and concurrent runs would make OOM, NaN, or disk-pressure
failures harder to attribute. After runs finish, read-only recomputation of
diagnostics and summary CSV generation from completed outputs is safe.

For regularized K-resolution sweeps, run each K serially on the GPU even though
each K writes to a separate output directory. The sweep is meant to compare
program resolution under identical resource and regularization settings, so
parallel GPU contention would make runtime failures and early-stopping behavior
harder to interpret. After all K runs finish, summary/recommendation generation
and read-only review of program reports can be parallelized.

For NMF warm-start experiments, keep train-split-only NMF fitting as a single
coordinated CPU job per dataset/K, then run each full260K bottleneck mode
serially on the GPU. The fixed, trainable, and anchor modes all consume the
same full dataset, embeddings, and NMF dictionary, so concurrent training would
make OOM, NaN, or disk-pressure failures harder to attribute. Once all modes
finish, read-only summary generation and review of dictionary drift,
reconstruction metrics, and top-taxa/biome reports can be parallelized.

For the curated experiment ledger at `docs/experiments.md`, read-only metric
extraction from immutable CSV/JSON artifacts can be parallelized. Edits to the
ledger itself should stay serialized through one coordinating agent, because it
is the canonical human-facing interpretation layer and must keep goals,
evidence tables, artifact links, and judgments internally consistent.

## Loop Discipline

Each execution loop should:

1. Record the intended command and expected artifact in `implementation-notes.md`.
2. Run the command.
3. Record failures verbatim enough to diagnose environment/data/code causes.
4. Patch the smallest necessary surface.
5. Re-run the same command before advancing.
