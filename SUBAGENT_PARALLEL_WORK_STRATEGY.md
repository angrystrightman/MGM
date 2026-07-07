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

## Loop Discipline

Each execution loop should:

1. Record the intended command and expected artifact in `implementation-notes.md`.
2. Run the command.
3. Record failures verbatim enough to diagnose environment/data/code causes.
4. Patch the smallest necessary surface.
5. Re-run the same command before advancing.
