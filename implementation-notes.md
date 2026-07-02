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
