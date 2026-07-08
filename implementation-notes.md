# Implementation Notes

## 2026-07-02 MGM Phased Reproduction Scaffold

- Created a dedicated local branch `codex/mgm-repro-scaffold` before editing
  files because the checkout was on `main` and no linked worktree was active.
- Chose a clean `mgm-repro` conda environment instead of reusing the existing
  `Histo-PEFT` environment. `Histo-PEFT` has newer torch/transformers versions
  and lacks `pytorch_lightning`, so it is a poor reproduction baseline.
- Kept `prepare_infant_smoke_data.py` dependency-free so data-subset creation
  can be tested before the MGM conda environment exists.
- Used C/V delivery labels as the fast smoke target and kept the original
  12-class `Env` labels for notebook-fidelity runs. The notebook prose says
  Cesarean vs. Vaginal, but the raw `Env` column encodes age plus delivery mode.
- Added resource checks before training because the repository does include a
  packaged `general_model/pytorch_model.bin`; the main uncertainty is runtime
  compatibility, not missing weights.
- Added a minimal `.gitignore` before running smoke commands so generated
  `runs/` artifacts and local build/cache output remain out of git.
- Kept generation/reconstruction outside the smoke success criteria. The
  generator path expects an extended tokenizer saved as `tokenizer.pkl`, which
  the packaged default language model does not provide.
- Full MicroCorpus-260K pretraining is deferred. The local corpus pickle is
  about 11GB and should be treated as a planned long run after the infant smoke
  loop proves the environment and model path.
- Attempted `conda env create -f environment-mgm.yml`; conda's pip phase hid
  progress while downloading the GPU-enabled `torch==2.0.1` wheel. A direct pip
  retry showed the 619.9MB wheel downloading at about 83KB/s, so using that as a
  hard blocker would stall the smoke workflow. For this scaffold, the practical
  fallback is CPU-only torch for Phase 0-5 smoke verification; GPU-enabled torch
  remains the formal long-run target.
- Added `environment-mgm-cpu.yml` to capture that fallback. The CPU wheel
  `torch==2.0.1+cpu` downloaded quickly from the PyTorch CPU index, then the
  rest of the pinned MGM stack installed normally.
- Set `PYTHONNOUSERSITE=1` in the smoke runner and installed
  `typing_extensions` inside the conda env so verification does not silently
  depend on packages from `~/.local`.
- Made the smoke runner clean its own default `runs/` output before execution.
  The cleanup refuses paths outside this repository's `runs/` directory, so
  accidental external run roots are not deleted.
- Verified the CPU fallback end-to-end with
  `ENV_NAME=mgm-repro MAX_SAMPLES=24 ./scripts/run_infant_smoke.sh`. The run
  selected 24 balanced C/V samples, then `mgm construct` dropped 1 all-zero
  sample, leaving 23 corpus samples. Finetune, predict/evaluate, and
  representation extraction completed on CPU.
- Representation verification found 16 extracted sample embeddings with width
  256. Attention top-k produced 147 rows rather than `16 * 10` because some
  samples have fewer than 10 non-special genus tokens; the accepted invariant is
  1 to `top_k` rows per extracted sample.

## 2026-07-02 GPU Fresh Verification Loop

- The user completed installation of the GPU PyPI wheel and its CUDA 11.7
  runtime dependencies in the existing `mgm-repro` conda environment. The
  environment is now the primary route for reproduction; `environment-mgm-cpu.yml`
  remains a fallback record rather than the active path.
- Verified before the fresh loop that `torch.__version__` reports
  `2.0.1+cu117`, `torch.cuda.is_available()` is `True`, CUDA runtime is `11.7`,
  and 8 RTX 4090 devices are visible.
- The old `runs/mgm_repro/infant_smoke` artifacts record `python_device: cpu` in
  `resource_check.json` and `device: cpu` in the representation manifest, so
  they are historical evidence only. The GPU acceptance loop must recreate
  `runs/mgm_repro/infant_smoke` with `CLEAN_RUN=1`.
- Fresh loop commands to run and record:

```bash
PYTHONNOUSERSITE=1 /home/sunyirong/miniforge3/envs/mgm-repro/bin/python -m pip check
PYTHONNOUSERSITE=1 /home/sunyirong/miniforge3/envs/mgm-repro/bin/python scripts/check_mgm_resources.py --device cuda
PYTHONNOUSERSITE=1 /home/sunyirong/miniforge3/envs/mgm-repro/bin/python -m unittest tests.test_mgm_repro_scripts -v
ENV_NAME=mgm-repro MAX_SAMPLES=96 ./scripts/run_infant_smoke.sh
```

- Expected artifacts are `resource_check.json`, `data/corpus.pkl`,
  `model/label_encoder.pkl`, `predictions/y_score.csv`,
  `predictions/evaluation/avg.csv`, `representations/sample_embeddings.csv`,
  and `representations/attention_topk.csv` under
  `runs/mgm_repro/infant_smoke`.
- First fresh GPU smoke attempt failed during `mgm finetune` with a
  segmentation fault. `PYTHONFAULTHANDLER=1` localized the crash to
  `torch.nn.parallel.data_parallel -> broadcast_coalesced` inside
  `transformers.Trainer`, meaning the Trainer saw multiple visible GPUs and
  wrapped the model with `DataParallel`.
- A manual CUDA forward/backward/AdamW step on the same corpus succeeded, and a
  one-step Trainer run succeeded when launched with `CUDA_VISIBLE_DEVICES=2`.
  The root cause is the multi-GPU DataParallel path, not the packaged model,
  corpus, labels, or single-GPU CUDA runtime.
- Updated `scripts/run_infant_smoke.sh` to default to one visible GPU through
  `SMOKE_CUDA_VISIBLE_DEVICES=0` while preserving an explicitly supplied
  `CUDA_VISIBLE_DEVICES`. This keeps smoke deterministic and lets long runs
  choose their own GPU policy.
- Re-ran `ENV_NAME=mgm-repro MAX_SAMPLES=96 ./scripts/run_infant_smoke.sh`
  after the single-GPU fix. The script printed `CUDA_VISIBLE_DEVICES=0`, resource
  check ran on CUDA, `construct` selected 96 samples and dropped 1 all-zero
  sample, `finetune` completed 18 training steps, `predict -E` produced
  evaluation output, and representation extraction wrote CUDA embeddings.
- Artifact validation passed: 95 corpus samples, 16 embedding rows, embedding
  width 256, 160 attention top-k rows, `resource_device=cuda`, and
  `representation_device=cuda`.
- `/home/sunyirong/shared/sunyirong/wheels` was empty when checked after GPU
  validation, so no wheel cleanup command was needed.
- The upstream repository tracks Python bytecode under `mgm/CLI/__pycache__`.
  Running Python 3.10 in this environment updated several tracked `.pyc` files;
  these are cache noise and not part of the reproduction scaffold. The smoke
  runner now exports `PYTHONDONTWRITEBYTECODE=1` to reduce future cache writes.
- After adding `PYTHONDONTWRITEBYTECODE=1`, the final verification loop was run
  again: `pip check`, CUDA resource check, 6 script unit tests, full
  `MAX_SAMPLES=96` smoke, and artifact validation all exited successfully.

## 2026-07-07 ComPASS-Biome Feasibility Layer

- Project goal shifted from MGM reproduction alone to a downstream
  interpretability module: frozen MGM teacher embeddings, soft community-program
  activations, taxa reconstruction, and metadata readouts.
- Local MicroCorpus-260K inspection found exactly three dataset files under
  `/home/sunyirong/shared/sunyirong/MicroCorpus-260K`:
  `MicroCorpus-260K.pkl`, `MicroCorpus-260K_unnorm.pkl`, and
  `mgnify_biomes.csv`.
- Both corpus pickle files load as `mgm.src.MicroCorpus.MicroCorpus` with
  263,302 samples, 512 token positions, a tokenizer vocabulary of 9,669, and
  sample x genus `.data` matrices with 9,665 taxa columns.
- The normalized corpus `.data` is dense z-scored abundance-like data. The
  unnormalized corpus `.data` is sparse relative abundance with rows summing to
  approximately 1.0, so it is the preferred first decoder target `X [N,G]`.
- `mgnify_biomes.csv` contains a unique sample/run accession column plus five
  MGnify biome hierarchy columns. It does not contain explicit disease,
  phenotype, study/cohort, country, sequencing platform, or pathway abundance
  fields.
- Alignment was exact in local inspection: 263,302 corpus samples, 263,302
  metadata rows, 263,302 matches, no unmatched rows, no duplicate sample IDs,
  and matching order.
- Added a ComPASS-Biome script layer for data audit, pilot dataset construction,
  batched teacher embedding extraction, and a lightweight bottleneck pilot
  baseline. Generated artifacts must stay under `runs/compass_biome/`.
- First validation target is taxa reconstruction plus biome/source
  interpretability. Disease/pathway validation is deferred to curated disease
  cohorts or datasets with pathway abundance.
- Implemented and verified the first smoke loop:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 /home/sunyirong/miniforge3/envs/mgm-repro/bin/python scripts/audit_microcorpus_260k.py
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 /home/sunyirong/miniforge3/envs/mgm-repro/bin/python scripts/build_compass_pilot_dataset.py --max-samples 128 --output-dir runs/compass_biome/pilot_dataset --label-column biome_1 --seed 0
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 /home/sunyirong/miniforge3/envs/mgm-repro/bin/python scripts/extract_compass_embeddings.py --sample-ids runs/compass_biome/pilot_dataset/sample_ids.csv --output-dir runs/compass_biome/embeddings --batch-size 16 --device auto
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 /home/sunyirong/miniforge3/envs/mgm-repro/bin/python scripts/run_compass_bottleneck_pilot.py --dataset-dir runs/compass_biome/pilot_dataset --embeddings runs/compass_biome/embeddings/embeddings.npz --output-dir runs/compass_biome/bottleneck_pilot --num-programs 8 --epochs 20 --batch-size 32 --learning-rate 0.01 --device auto --top-k 20 --label-column biome_1
```

- Artifact validation passed: audit samples 263,302; pilot `X_unnorm` shape
  `(128, 9665)`; embedding shape `(128, 256)`; activation matrix `A` shape
  `(128, 8)`; taxa dictionary `P_taxa` shape `(8, 9665)`; bottleneck loss
  decreased from 9.1114 to 7.8759. This is enough to proceed with method
  design, but it is not a biological claim yet.

## 2026-07-07 ComPASS-Biome Taxa Readout Sanity Check

- Upgraded the ComPASS pilot from a same-sample loss-decrease smoke test to a
  split-aware taxa readout sanity check. The dataset builder now writes
  `sample_splits.csv`, `taxa_filter.csv`, and `X_taxa.npz`.
- Top taxa are selected from the train split only, using train prevalence with
  train mean abundance as the tie-break. The decoder target is the selected taxa
  abundance renormalized within the Top1000 taxa universe, and retained mass is
  written so metric interpretation stays explicit.
- A 10k dataset build exposed six samples with zero retained mass after Top1000
  filtering. These samples have no valid filtered taxa distribution, so the
  builder now drops zero-retained samples and records
  `dropped_zero_retained_samples` in the manifest.
- Added split-level reconstruction metrics: cross entropy, Top-20 taxa recall,
  and Bray-Curtis similarity. Added two controls: train-mean abundance baseline
  and shuffled train Z-X pairing.
- Added program diagnostics for collapse/redundancy: dead program fraction,
  effective programs per sample, activation entropy, dictionary entropy, and
  off-diagonal dictionary cosine similarity.
- Added PNG reports with a hard `matplotlib==3.7.5` dependency:
  `program_usage_heatmap.png`, `example_true_vs_reconstructed_taxa.png`,
  `top_taxa_per_program.png`, and `metrics_comparison.png`.
- Verification commands run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 /home/sunyirong/miniforge3/envs/mgm-repro/bin/python -m unittest tests.test_compass_biome_scripts -v
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 /home/sunyirong/miniforge3/envs/mgm-repro/bin/python -m py_compile scripts/compass_biome_utils.py scripts/build_compass_pilot_dataset.py scripts/run_compass_bottleneck_pilot.py scripts/extract_compass_embeddings.py
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 /home/sunyirong/miniforge3/envs/mgm-repro/bin/python -m pip check
```

- The refreshed 1k smoke used `N=1024`, `G=1000`, `K=16`, 20 epochs, and
  `CUDA_VISIBLE_DEVICES=0`. It produced all required CSV/PNG/NPZ artifacts. Test
  metrics were: real Top-20 recall 0.2814, real Bray-Curtis 0.1211; mean
  baseline Top-20 recall 0.1644, Bray-Curtis 0.1206; shuffle Top-20 recall
  0.1757, Bray-Curtis 0.0960.
- The 1k smoke showed heavy program collapse: dead program fraction 0.8125 and
  effective programs per sample 1.1091. This is acceptable as a smoke result
  but not yet an interpretable-program success.
- The 10k validation used `max_samples=10000`, dropped 6 zero-retained samples,
  and trained on 9,994 samples with split counts train 7,994, valid 1,000, test
  1,000. Mean retained mass was 0.9263.
- The 10k validation test metrics were strongly positive for taxa reconstruction
  signal: real Top-20 recall 0.3838 and Bray-Curtis 0.2877 versus mean baseline
  Top-20 recall 0.1870 and Bray-Curtis 0.1237, and shuffle Top-20 recall 0.1870
  and Bray-Curtis 0.1235.
- The 10k validation still showed program collapse: dead program fraction 0.75,
  effective programs per sample 1.3906, and dictionary off-diagonal cosine
  0.6342. Current conclusion is positive for MGM-embedding taxa readout, but
  only medium for interpretable program discovery. The next modeling iteration
  should add anti-collapse pressure such as activation entropy/usage balancing,
  dictionary diversity, sparsity tuning, K sweep, or NMF/cvaNMF warm start.

## 2026-07-07 ComPASS-Biome 100K Taxa Readout Validation

- Started the 100K scale-up to test whether the 10K taxa readout improvement is
  stable at larger sample count. Baseline verification before launch: 27 unit
  tests passed, all 8 RTX 4090 GPUs were idle, host RAM had about 983GiB
  available, and `/data` had about 39GB free.
- Kept the comparison intentionally single-setting rather than a hyperparameter
  sweep: `max_samples=100000`, `num_taxa=1000`, `K=32`,
  `batch_size=1024`, `learning_rate=0.02`, `epochs=100`, `patience=15`, and
  `controls=all`. The learning rate was increased with the larger batch, but
  kept below a full linear scaling jump from the 10K run.
- Dataset build completed with 99,924 usable samples after dropping 76
  zero-retained samples. Split counts were train 79,948, valid 9,988, and test
  9,988. `X_taxa` shape was `(99924, 1000)`, row sums were near 1, and mean
  retained Top1000 mass was 0.9267.
- Frozen MGM embedding extraction completed on CUDA with batch size 64.
  Embeddings had shape `(99924, 256)` and all values were finite.
- Bottleneck validation completed with `K=32`, `batch_size=1024`,
  `learning_rate=0.02`, and `patience=15`. Early stopping restored the best
  validation checkpoint after 76 real-model epochs. The run wrote
  `runs/compass_biome/taxa_readout_validation_100k/bottleneck`.
- 100K test metrics were: real Top-20 recall 0.4261 and Bray-Curtis 0.3215;
  mean baseline Top-20 recall 0.1749 and Bray-Curtis 0.1138; shuffle Top-20
  recall 0.1749 and Bray-Curtis 0.1136.
- Compared with the 10K run, the real test signal improved from Top-20 recall
  0.3838 to 0.4261 and Bray-Curtis 0.2877 to 0.3215. Controls remained near
  their 10K level, so the larger sample count strengthens the conclusion that
  frozen MGM embeddings carry taxa-composition information.
- Program collapse improved slightly but remains the main limitation:
  dead program fraction 0.7188, effective programs per sample 1.5176,
  activation entropy mean 0.3371, dictionary entropy mean 5.9359, and
  dictionary off-diagonal cosine 0.4774. This is still not enough to claim
  well-spread interpretable programs.
- Artifact validation passed: `X_taxa (99924,1000)`, embeddings `(99924,256)`,
  `A (99924,32)`, `P_taxa (32,1000)`, reconstruction `(99924,1000)`, aligned
  sample IDs, finite arrays, positive retained mass, and real test metrics above
  both controls.

## 2026-07-07 ComPASS-Biome Full 260K Taxa Readout Validation

- Started the full MicroCorpus-260K scale-up to test whether the 100K taxa
  readout signal is stable on all usable samples. Baseline verification before
  launch: 27 unit tests passed, `py_compile` passed for ComPASS scripts,
  `pip check` reported no broken requirements, all 8 RTX 4090 GPUs were idle,
  host RAM had about 983GiB available, and `/data` had about 38GB free.
- Kept this as a single-setting scale test rather than a hyperparameter sweep:
  `max_samples=263302`, `num_taxa=1000`, `K=32`, embedding batch size 64,
  bottleneck `batch_size=2048`, `learning_rate=0.03`, `epochs=100`,
  `patience=15`, and `controls=all`. Fallback policy is embedding batch size 32
  or bottleneck `batch_size=1024`, `learning_rate=0.02` only if the primary run
  hits OOM or non-finite values.
- Dataset build completed with 263,065 usable samples after dropping 235
  zero-retained samples. The two rows with missing `biome_1` are excluded by the
  current balanced-by-biome selection semantics before retained-mass filtering.
  Split counts were train 210,459, valid 26,306, and test 26,300. `X_taxa`
  shape was `(263065, 1000)`, row sums were near 1, and mean retained Top1000
  mass was 0.9461.
- Frozen MGM embedding extraction completed on CUDA with batch size 64.
  Embeddings had shape `(263065, 256)` and all values were finite.
- Bottleneck validation completed with the primary setting: `K=32`,
  `batch_size=2048`, `learning_rate=0.03`, `patience=15`, and `controls=all`.
  Early stopping restored the best validation checkpoint after 54 real-model
  epochs, so the fallback setting was not needed.
- Full260K test metrics were: real Top-20 recall 0.4543 and Bray-Curtis 0.3306;
  mean baseline Top-20 recall 0.2187 and Bray-Curtis 0.1249; shuffle Top-20
  recall 0.2189 and Bray-Curtis 0.1248.
- Compared with 100K, full260K improved real Top-20 recall from 0.4261 to
  0.4543 and Bray-Curtis from 0.3215 to 0.3306. Controls also rose in Top-20
  recall because the full train distribution includes more common taxa, but
  real remains clearly separated from both controls.
- Program usage is still the limiting factor: dead program fraction 0.75,
  effective programs per sample 1.4863, activation entropy mean 0.3247,
  dictionary entropy mean 5.9809, and dictionary off-diagonal cosine 0.5176.
  This confirms the scale-up strengthens taxa readout, but does not solve
  interpretable program spread.
- Artifact validation passed: `X_taxa (263065,1000)`, embeddings
  `(263065,256)`, `A (263065,32)`, `P_taxa (32,1000)`, reconstruction
  `(263065,1000)`, aligned sample IDs, finite arrays, positive retained mass,
  real metrics above both controls, and real metrics above the 100K reference
  thresholds. The full260K run directory uses about 2.2GB.

## 2026-07-07 ComPASS-Biome Full260K K=16/8 Rerun

- Goal: rerun only the bottleneck stage on the existing full260K dataset and
  existing frozen MGM embeddings, changing only the program count from the
  baseline `K=32` to `K=16` and `K=8`.
- Keep all other settings matched to the full260K `K=32` run:
  `batch_size=2048`, `learning_rate=0.03`, `epochs=100`, `patience=15`,
  `controls=all`, `top_k=20`, `seed=0`, and `CUDA_VISIBLE_DEVICES=0`.
- Do not rebuild the dataset, do not re-extract embeddings, do not restore the
  deleted smoke/10k/100k result artifacts, and do not change MGM or ComPASS
  core model code.
- Preflight status: current branch is `codex/mgm-repro-scaffold`; the deleted
  tracked files are the user-removed smoke/10k/100k lightweight result
  artifacts; full260K dataset, embeddings, and `K=32` baseline metrics still
  exist; `/data` has about 35GB available; all 8 RTX 4090 GPUs are idle.
- Fresh preflight verification passed:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /home/sunyirong/miniforge3/envs/mgm-repro/bin/python \
  -m unittest tests.test_mgm_repro_scripts tests.test_compass_biome_scripts -v
```

- Planned `K=16` command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/sunyirong/miniforge3/envs/mgm-repro/bin/python \
  scripts/run_compass_bottleneck_pilot.py \
  --dataset-dir runs/compass_biome/taxa_readout_validation_full260k/dataset \
  --embeddings runs/compass_biome/taxa_readout_validation_full260k/embeddings/embeddings.npz \
  --output-dir runs/compass_biome/taxa_readout_validation_full260k_K16/bottleneck \
  --num-programs 16 \
  --epochs 100 \
  --batch-size 2048 \
  --learning-rate 0.03 \
  --controls all \
  --top-k 20 \
  --device auto \
  --patience 15
```

- Planned `K=8` command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/sunyirong/miniforge3/envs/mgm-repro/bin/python \
  scripts/run_compass_bottleneck_pilot.py \
  --dataset-dir runs/compass_biome/taxa_readout_validation_full260k/dataset \
  --embeddings runs/compass_biome/taxa_readout_validation_full260k/embeddings/embeddings.npz \
  --output-dir runs/compass_biome/taxa_readout_validation_full260k_K8/bottleneck \
  --num-programs 8 \
  --epochs 100 \
  --batch-size 2048 \
  --learning-rate 0.03 \
  --controls all \
  --top-k 20 \
  --device auto \
  --patience 15
```

- `K=16` run completed and artifact validation passed. Manifest recorded
  `num_samples=263065`, `num_taxa=1000`, `embedding_width=256`,
  `num_programs=16`, `epochs_ran=59`, `batch_size=2048`,
  `learning_rate=0.03`, and `device=cuda`.
- `K=16` arrays were finite with expected shapes: `A (263065,16)`,
  `P_taxa (16,1000)`, and reconstructions `(263065,1000)`. Required CSV and
  PNG report files were present and non-empty.
- `K=16` test metrics: real Top-20 recall 0.3241 and Bray-Curtis 0.2138.
  Program diagnostics worsened relative to `K=32`: dead program fraction
  0.8125, effective programs per sample 1.1502, activation entropy mean 0.1174,
  and dictionary off-diagonal cosine 0.6637.
- `K=8` run completed and artifact validation passed. Manifest recorded
  `num_samples=263065`, `num_taxa=1000`, `embedding_width=256`,
  `num_programs=8`, `epochs_ran=46`, `batch_size=2048`,
  `learning_rate=0.03`, and `device=cuda`.
- `K=8` arrays were finite with expected shapes: `A (263065,8)`,
  `P_taxa (8,1000)`, and reconstructions `(263065,1000)`. Required CSV and
  PNG report files were present and non-empty, and output sample IDs matched
  the reused full260K dataset order exactly.
- `K=8` test metrics: real Top-20 recall 0.4017 and Bray-Curtis 0.2705, versus
  mean baseline 0.2187/0.1249 and shuffle 0.2151/0.1250. Dead program fraction
  improved to 0.5000, effective programs per sample was 1.2235, activation
  entropy mean was 0.1665, and dictionary off-diagonal cosine improved to
  0.1956.
- Wrote the comparison artifact
  `runs/compass_biome/taxa_readout_validation_full260k_k_sweep_summary.csv`.
  Current interpretation: `K=32` remains best for reconstruction; `K=8` is a
  compact reconstruction tradeoff that still clearly beats controls and reduces
  dead-program fraction/dictionary redundancy; `K=16` is dominated by both
  `K=32` and `K=8` under this fixed hyperparameter setting.

## 2026-07-07 ComPASS-Biome Program Usage Balancing

- Goal: test a minimal anti-collapse regularization on the best-reconstructing
  full260K `K=32` bottleneck without changing MGM, the dataset, embeddings,
  controls, or primary reconstruction metrics.
- Design choice: add optional global usage balancing plus a local entropy
  target. Direct entropy minimization was not used because current
  `effective_programs_mean` is already low; minimizing entropy alone would push
  samples closer to one-hot usage.
- Implemented optional CLI arguments with default-off behavior:
  `--usage-balance-weight`, `--usage-balance-loss`,
  `--sample-entropy-weight`, and `--sample-entropy-target-effective`.
- Training metrics now include reconstruction and regularization components:
  `reconstruction_loss`, `regularized_loss`, `usage_balance_loss`,
  `sample_entropy_mean`, and `sample_entropy_target_loss`.
- TDD loop: added tests for collapsed-vs-uniform usage balance, entropy target,
  regularized training metrics, and manifest fields. The first targeted test
  run failed because `sample_entropy_target_loss` did not exist yet, then passed
  after the implementation.
- Fresh code readiness verification passed:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /home/sunyirong/miniforge3/envs/mgm-repro/bin/python \
  -m unittest tests.test_mgm_repro_scripts tests.test_compass_biome_scripts -v

PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /home/sunyirong/miniforge3/envs/mgm-repro/bin/python \
  -m py_compile scripts/compass_biome_utils.py scripts/run_compass_bottleneck_pilot.py

PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /home/sunyirong/miniforge3/envs/mgm-repro/bin/python -m pip check
```

- Planned small smoke command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/sunyirong/miniforge3/envs/mgm-repro/bin/python \
  scripts/run_compass_bottleneck_pilot.py \
  --dataset-dir runs/compass_biome/pilot_dataset \
  --embeddings runs/compass_biome/embeddings/embeddings.npz \
  --output-dir runs/compass_biome/usage_balance_smoke_K8/bottleneck \
  --num-programs 8 \
  --epochs 10 \
  --batch-size 32 \
  --learning-rate 0.01 \
  --controls none \
  --top-k 20 \
  --device auto \
  --usage-balance-weight 0.05 \
  --usage-balance-loss kl_uniform_to_usage \
  --sample-entropy-weight 0.10 \
  --sample-entropy-target-effective 2.0 \
  --no-plots
```

- Small usage-balance smoke completed on CUDA and passed artifact validation:
  `A (128,8)`, `P_taxa (8,9665)`, reconstructions `(128,9665)`, manifest
  records the new regularization arguments, and `training_metrics.csv` contains
  finite regularization columns. Final smoke metrics included reconstruction
  loss 8.5930, regularized loss 8.6044, usage balance loss 0.1348, mean sample
  entropy 0.6690, and entropy-target loss 0.0464.
- Planned full260K primary command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/sunyirong/miniforge3/envs/mgm-repro/bin/python \
  scripts/run_compass_bottleneck_pilot.py \
  --dataset-dir runs/compass_biome/taxa_readout_validation_full260k/dataset \
  --embeddings runs/compass_biome/taxa_readout_validation_full260k/embeddings/embeddings.npz \
  --output-dir runs/compass_biome/taxa_readout_validation_full260k_K32_balance005_entropy2/bottleneck \
  --num-programs 32 \
  --epochs 100 \
  --batch-size 2048 \
  --learning-rate 0.03 \
  --controls all \
  --top-k 20 \
  --device auto \
  --patience 15 \
  --usage-balance-weight 0.05 \
  --usage-balance-loss kl_uniform_to_usage \
  --sample-entropy-weight 0.10 \
  --sample-entropy-target-effective 2.0
```

- Full260K primary usage-balance run completed and passed artifact validation:
  manifest recorded `num_samples=263065`, `num_taxa=1000`,
  `embedding_width=256`, `num_programs=32`, `epochs_ran=54`,
  `usage_balance_weight=0.05`, `sample_entropy_weight=0.10`,
  `sample_entropy_target_effective=2.0`, and `device=cuda`.
- Output arrays were finite with expected shapes: `A (263065,32)`,
  `P_taxa (32,1000)`, and reconstructions `(263065,1000)`. Output sample IDs
  matched the full260K dataset order exactly.
- Full260K usage-balance test metrics were strongly positive: real Top-20
  recall 0.5422 and Bray-Curtis 0.4414, versus mean baseline 0.2187/0.1249 and
  shuffle 0.2162/0.1266.
- Program usage improved substantially relative to baseline `K=32`: dead
  program fraction 0.1563, effective programs per sample 2.1754, activation
  entropy mean 0.6540, and dictionary off-diagonal cosine 0.0289.
- Gate decision: do not run the stronger `balance010` condition. The primary
  run already reached `dead_program_fraction <= 0.50`, improved reconstruction
  above the unregularized baseline, and remained clearly separated from both
  controls. Running a second stronger setting now would be an unnecessary
  hyperparameter expansion.
- Wrote comparison artifact
  `runs/compass_biome/taxa_readout_validation_full260k_usage_balance_summary.csv`
  comparing baseline `K=32`, compact `K=8`, and regularized `K=32`.
