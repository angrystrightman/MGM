# MGM Reproduction Plan

This file tracks the active phased reproduction workflow for MGM. It does not
store checkpoints, generated corpora, or private data.

## Goal

Bring the local MGM repository to a reproducible state where we can train,
predict, and extract representations before designing downstream
representation-compression modules.

The next project layer is ComPASS-Biome: keep MGM frozen as a teacher, extract
sample embeddings, and test whether a small interpretable bottleneck can learn
community-program activations that reconstruct taxa and associate with metadata.

## Current Supported Capabilities

- `mgm construct`: converts abundance tables into ranked `MicroCorpus` pickle
  files.
- `mgm pretrain`: causal language-model pretraining from a corpus; optional
  label-token generator training.
- `mgm train`: supervised GPT-2 sequence classifier from scratch.
- `mgm finetune`: downstream classifier initialized from the packaged
  MicroCorpus-pretrained model.
- `mgm predict`: prediction and optional evaluation for finetuned classifiers.
- `mgm generate` and `mgm reconstruct`: experimental generation/reconstruction
  paths. `generate` expects a generator checkpoint with `tokenizer.pkl`, so the
  packaged `general_model` is not enough by itself.
- Representation analysis is available through notebook logic and now through
  `scripts/extract_mgm_representations.py`.

## Phased Workflow

### Phase 0: Environment

Create and verify the clean MGM GPU environment:

```bash
/home/sunyirong/miniforge3/bin/conda env create -f environment-mgm.yml
PYTHONNOUSERSITE=1 /home/sunyirong/miniforge3/envs/mgm-repro/bin/python -m pip check
PYTHONNOUSERSITE=1 /home/sunyirong/miniforge3/envs/mgm-repro/bin/python -c "import torch, transformers, pandas, mgm; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda, transformers.__version__)"
```

Current local status: `mgm-repro` has GPU-enabled `torch==2.0.1+cu117`,
CUDA 11.7 runtime packages, `torch.cuda.is_available() == True`, and 8 visible
RTX 4090 devices.

Keep the CPU environment file only as a fallback when GPU package installation
is unavailable:

```bash
/home/sunyirong/miniforge3/bin/conda env create -f environment-mgm-cpu.yml
/home/sunyirong/miniforge3/bin/conda run -n mgm-repro python -m pip install -e . --no-deps
```

Termination condition: Python imports MGM dependencies, `pip check` reports no
broken requirements, `torch==2.0.1+cu117`, CUDA is available, and a small GPU
tensor operation returns finite values.

### Phase 1: Packaged Resources

```bash
/home/sunyirong/miniforge3/bin/conda run -n mgm-repro \
  python scripts/check_mgm_resources.py \
  --device cuda \
  --json-output runs/mgm_repro/resource_check.json
```

Termination condition: tokenizer, config, generation config, and
`pytorch_model.bin` exist; a tiny forward pass returns finite logits, hidden
states, and attentions on CUDA.

### Phase 2: Infant Smoke Data

```bash
/home/sunyirong/miniforge3/bin/conda run -n mgm-repro \
  python scripts/prepare_infant_smoke_data.py \
  --output-dir runs/mgm_repro/infant_smoke/data \
  --max-samples 96 \
  --seed 0
```

Termination condition: subset abundance, 12-class `Env` labels, C/V labels, and
manifest exist and sample IDs align.

### Phase 3-5: End-to-End Smoke

```bash
ENV_NAME=mgm-repro MAX_SAMPLES=96 ./scripts/run_infant_smoke.sh
```

For the CPU fallback environment, use a smaller first pass:

```bash
ENV_NAME=mgm-repro MAX_SAMPLES=24 ./scripts/run_infant_smoke.sh
```

The smoke runner defaults to `CLEAN_RUN=1` and will only remove output under
this repository's `runs/` directory. Set `CLEAN_RUN=0` to inspect or reuse a
previous run directory.

The smoke runner also defaults to single-GPU visibility
(`SMOKE_CUDA_VISIBLE_DEVICES=0`) unless `CUDA_VISIBLE_DEVICES` is already set.
This avoids Hugging Face `Trainer` automatically wrapping the model in
`torch.nn.DataParallel` across all visible GPUs during the smoke test. Override
with `SMOKE_CUDA_VISIBLE_DEVICES=<gpu_index>` or explicit
`CUDA_VISIBLE_DEVICES=<list>` when needed.

The prior `runs/mgm_repro/infant_smoke` artifact was produced by the CPU
fallback run, so GPU acceptance requires a fresh run after CUDA installation.

Termination condition:

- `runs/mgm_repro/infant_smoke/data/corpus.pkl` exists.
- `runs/mgm_repro/infant_smoke/model/label_encoder.pkl` exists.
- `runs/mgm_repro/infant_smoke/predictions/y_score.csv` exists.
- `runs/mgm_repro/infant_smoke/predictions/evaluation/avg.csv` exists.
- `sample_embeddings.csv`, `attention_topk.csv`, and representation manifest
  exist with aligned sample IDs.
- Attention output contains up to `top_k` rows per sample. Some infant smoke
  samples have fewer than 10 non-special genus tokens after preprocessing.

Latest fresh GPU verification on 2026-07-02:

- `MAX_SAMPLES=96` selected 96 infant samples; `construct` dropped 1 all-zero
  sample, leaving 95 corpus samples.
- `finetune`, `predict -E`, and representation extraction completed with
  `CUDA_VISIBLE_DEVICES=0`.
- `resource_check.json` records `python_device: cuda`; representation manifest
  records `device: cuda`, `num_samples: 16`, and `embedding_width: 256`.
- Artifact validation confirmed required files are non-empty, embedding sample
  IDs align to the first 16 corpus samples, and embedding/attention values are
  finite.

### Phase 6: MicroCorpus-260K Long-Run Preparation

Known local path:

```text
/home/sunyirong/shared/sunyirong/MicroCorpus-260K
```

Observed files:

- `MicroCorpus-260K.pkl`: about 11GB.
- `mgnify_biomes.csv`: metadata for 263,302 samples.

Long-run pretraining command template:

```bash
tmux new -s mgm_pretrain_260k
conda activate mgm-repro
cd /data/shared/sunyirong/workspace/code/MGM
CUDA_VISIBLE_DEVICES=<free_gpu> mgm pretrain \
  -i /home/sunyirong/shared/sunyirong/MicroCorpus-260K/MicroCorpus-260K.pkl \
  -o /home/sunyirong/shared/sunyirong/MGM-runs/pretrain_260k \
  -H /home/sunyirong/shared/sunyirong/MGM-runs/pretrain_260k_logs \
  --seed 0
```

Do not start full pretraining until the infant smoke workflow has passed and a
GPU/runtime budget is explicitly chosen.

### Phase 7: ComPASS-Biome Data Audit

Inspect MicroCorpus-260K without assuming README claims:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /home/sunyirong/miniforge3/envs/mgm-repro/bin/python \
  scripts/audit_microcorpus_260k.py
```

Termination condition:

- File inventory covers `MicroCorpus-260K.pkl`, `MicroCorpus-260K_unnorm.pkl`,
  and `mgnify_biomes.csv`.
- Both pickle files load as `mgm.src.MicroCorpus.MicroCorpus`.
- Both corpora expose `tokens`, `tokenizer`, and sample x genus `.data`.
- Metadata aligns exactly to corpus sample IDs.
- The report explicitly marks disease, pathway, cohort, country, and platform
  validation as unsupported unless matching columns are found.

### Phase 8: ComPASS-Biome Pilot Dataset

Build an output-isolated pilot subset:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /home/sunyirong/miniforge3/envs/mgm-repro/bin/python \
  scripts/build_compass_pilot_dataset.py \
  --max-samples 128 \
  --output-dir runs/compass_biome/pilot_dataset
```

Termination condition:

- `sample_ids.csv`, `metadata.csv`, `taxa_names.csv`, `X_unnorm.npz`, and
  `manifest.json` exist.
- `X_unnorm.npz` is sample x genus, taxa order is fixed, and row sums are near
  1.0.
- Selected IDs exist in teacher corpus, unnormalized corpus, and metadata.

### Phase 9: ComPASS-Biome Embedding And Bottleneck Smoke

Extract frozen MGM embeddings and train a small bottleneck:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/sunyirong/miniforge3/envs/mgm-repro/bin/python \
  scripts/extract_compass_embeddings.py \
  --sample-ids runs/compass_biome/pilot_dataset/sample_ids.csv \
  --output-dir runs/compass_biome/embeddings \
  --batch-size 16

PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/sunyirong/miniforge3/envs/mgm-repro/bin/python \
  scripts/run_compass_bottleneck_pilot.py \
  --dataset-dir runs/compass_biome/pilot_dataset \
  --embeddings runs/compass_biome/embeddings/embeddings.npz \
  --output-dir runs/compass_biome/bottleneck_pilot \
  --num-programs 8 \
  --epochs 20 \
  --batch-size 32
```

Termination condition:

- Embeddings are finite, sample IDs align, and width is 256.
- Bottleneck outputs contain `A [N,K]`, `P_taxa [K,G]`, metrics, top taxa, and
  biome enrichment.
- Reconstruction loss decreases in the pilot run.
- Results are treated as feasibility evidence, not a biological claim.

Latest ComPASS-Biome smoke verification on 2026-07-07:

- Audit confirmed 263,302 samples, 9,665 taxa columns, tokenizer vocab size
  9,669, exact corpus/metadata alignment, and no disease/pathway/cohort/country
  or platform fields.
- Pilot dataset used 128 samples balanced by `biome_1`; `X_unnorm` shape was
  `(128, 9665)` with row sums near 1.0.
- Frozen MGM teacher embeddings were extracted on CUDA with shape `(128, 256)`.
- Bottleneck smoke used `K=8`, produced `A (128, 8)` and
  `P_taxa (8, 9665)`, and reconstruction loss decreased from 9.1114 to 7.8759
  over 20 epochs.

### Phase 10: ComPASS-Biome Taxa Readout Sanity Check

Run the first method-level validation without disease/pathway/cohort claims:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /home/sunyirong/miniforge3/envs/mgm-repro/bin/python \
  scripts/build_compass_pilot_dataset.py \
  --max-samples 1024 --num-taxa 1000 --seed 0 \
  --output-dir runs/compass_biome/taxa_readout_smoke/dataset

PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/sunyirong/miniforge3/envs/mgm-repro/bin/python \
  scripts/extract_compass_embeddings.py \
  --sample-ids runs/compass_biome/taxa_readout_smoke/dataset/sample_ids.csv \
  --output-dir runs/compass_biome/taxa_readout_smoke/embeddings \
  --batch-size 32

PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/sunyirong/miniforge3/envs/mgm-repro/bin/python \
  scripts/run_compass_bottleneck_pilot.py \
  --dataset-dir runs/compass_biome/taxa_readout_smoke/dataset \
  --embeddings runs/compass_biome/taxa_readout_smoke/embeddings/embeddings.npz \
  --output-dir runs/compass_biome/taxa_readout_smoke/bottleneck \
  --num-programs 16 --epochs 20 --batch-size 128 \
  --controls all --top-k 20 --device auto
```

For the first larger validation:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /home/sunyirong/miniforge3/envs/mgm-repro/bin/python \
  scripts/build_compass_pilot_dataset.py \
  --max-samples 10000 --num-taxa 1000 --seed 0 \
  --output-dir runs/compass_biome/taxa_readout_validation_10k/dataset

PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/sunyirong/miniforge3/envs/mgm-repro/bin/python \
  scripts/extract_compass_embeddings.py \
  --sample-ids runs/compass_biome/taxa_readout_validation_10k/dataset/sample_ids.csv \
  --output-dir runs/compass_biome/taxa_readout_validation_10k/embeddings \
  --batch-size 32

PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/sunyirong/miniforge3/envs/mgm-repro/bin/python \
  scripts/run_compass_bottleneck_pilot.py \
  --dataset-dir runs/compass_biome/taxa_readout_validation_10k/dataset \
  --embeddings runs/compass_biome/taxa_readout_validation_10k/embeddings/embeddings.npz \
  --output-dir runs/compass_biome/taxa_readout_validation_10k/bottleneck \
  --num-programs 32 --epochs 100 --batch-size 256 \
  --controls all --top-k 20 --device auto --patience 15
```

For the 100K scale-up validation:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /home/sunyirong/miniforge3/envs/mgm-repro/bin/python \
  scripts/build_compass_pilot_dataset.py \
  --max-samples 100000 --num-taxa 1000 --seed 0 \
  --output-dir runs/compass_biome/taxa_readout_validation_100k/dataset

PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/sunyirong/miniforge3/envs/mgm-repro/bin/python \
  scripts/extract_compass_embeddings.py \
  --sample-ids runs/compass_biome/taxa_readout_validation_100k/dataset/sample_ids.csv \
  --output-dir runs/compass_biome/taxa_readout_validation_100k/embeddings \
  --batch-size 64

PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/sunyirong/miniforge3/envs/mgm-repro/bin/python \
  scripts/run_compass_bottleneck_pilot.py \
  --dataset-dir runs/compass_biome/taxa_readout_validation_100k/dataset \
  --embeddings runs/compass_biome/taxa_readout_validation_100k/embeddings/embeddings.npz \
  --output-dir runs/compass_biome/taxa_readout_validation_100k/bottleneck \
  --num-programs 32 --epochs 100 --batch-size 1024 \
  --learning-rate 0.02 --controls all --top-k 20 \
  --device auto --patience 15
```

For the full MicroCorpus-260K scale-up validation:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /home/sunyirong/miniforge3/envs/mgm-repro/bin/python \
  scripts/build_compass_pilot_dataset.py \
  --max-samples 263302 --num-taxa 1000 --seed 0 \
  --output-dir runs/compass_biome/taxa_readout_validation_full260k/dataset

PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/sunyirong/miniforge3/envs/mgm-repro/bin/python \
  scripts/extract_compass_embeddings.py \
  --sample-ids runs/compass_biome/taxa_readout_validation_full260k/dataset/sample_ids.csv \
  --output-dir runs/compass_biome/taxa_readout_validation_full260k/embeddings \
  --batch-size 64

PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/sunyirong/miniforge3/envs/mgm-repro/bin/python \
  scripts/run_compass_bottleneck_pilot.py \
  --dataset-dir runs/compass_biome/taxa_readout_validation_full260k/dataset \
  --embeddings runs/compass_biome/taxa_readout_validation_full260k/embeddings/embeddings.npz \
  --output-dir runs/compass_biome/taxa_readout_validation_full260k/bottleneck \
  --num-programs 32 --epochs 100 --batch-size 2048 \
  --learning-rate 0.03 --controls all --top-k 20 \
  --device auto --patience 15
```

Termination condition:

- Dataset artifacts include `sample_ids.csv`, `sample_splits.csv`,
  `X_taxa.npz`, `taxa_filter.csv`, and `manifest.json`.
- Top1000 taxa are selected from train split only; zero-retained samples are
  dropped and counted in the manifest.
- Embeddings are finite, sample IDs align, and width is 256.
- Bottleneck outputs include `A [N,K]`, `P_taxa [K,G]`, reconstructions,
  split-level reconstruction metrics, program diagnostics, top taxa,
  biome enrichment, and PNG reports.
- `reconstruction_metrics.csv` contains real, shuffle, and mean-baseline rows
  for train, valid, and test splits.

Latest validation on 2026-07-07:

- 1k smoke: `N=1024`, `G=1000`, `K=16`; real test Top-20 recall 0.2814 and
  Bray-Curtis 0.1211. Top-20 recall beat mean/shuffle controls, but
  Bray-Curtis was close to mean baseline. Program usage collapsed heavily
  (`dead_program_fraction=0.8125`).
- 10k validation: requested 10,000 samples; 6 zero-retained samples were
  dropped, leaving `N=9994`, `G=1000`, `K=32`. Real test Top-20 recall was
  0.3838 and Bray-Curtis 0.2877, versus mean baseline 0.1870/0.1237 and
  shuffle 0.1870/0.1235.
- 100k validation: requested 100,000 samples; 76 zero-retained samples were
  dropped, leaving `N=99924`, `G=1000`, `K=32`. Real test Top-20 recall was
  0.4261 and Bray-Curtis 0.3215, versus mean baseline 0.1749/0.1138 and
  shuffle 0.1749/0.1136. Early stopping restored the best validation checkpoint
  after 76 real-model epochs.
- Full260K validation: requested all 263,302 rows; the current biome-stratified
  selection excludes 2 missing `biome_1` rows, and 235 zero-retained samples
  were dropped, leaving `N=263065`, `G=1000`, `K=32`. Real test Top-20 recall
  was 0.4543 and Bray-Curtis 0.3306, versus mean baseline 0.2187/0.1249 and
  shuffle 0.2189/0.1248. Early stopping restored the best validation checkpoint
  after 54 real-model epochs.
- Interpretation: the taxa readout sanity check is strongly positive for MGM
  embedding carrying taxa-composition signal, and the signal improved from 10k
  to 100k to full260K. It is still only medium for interpretable program
  discovery because usage remains collapsed (`dead_program_fraction=0.75`,
  effective programs per sample 1.4863). Next modeling work should add
  anti-collapse regularization or warm starts before making biological claims.
