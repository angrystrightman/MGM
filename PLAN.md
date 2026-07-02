# MGM Reproduction Plan

This file tracks the active phased reproduction workflow for MGM. It does not
store checkpoints, generated corpora, or private data.

## Goal

Bring the local MGM repository to a reproducible state where we can train,
predict, and extract representations before designing downstream
representation-compression modules.

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
