# SKCQ — Spherical K-means Codebook Quantization

Post-training quantization of Mixture-of-Experts (MoE) expert weights for
**Qwen3.6-35B-A3B**, using spherical product-quantization codebooks with
additive real-error residuals. The goal is to push expert weights toward
~1–4 bits/weight while keeping perplexity/KL divergence close to the bf16
baseline, and to *sweep* the configuration space to find the Pareto frontier of
(bits-per-weight, reconstruction error) per layer.

> This is a research repo. It targets one specific model and a specific
> multi-GPU lab setup (see [Hardware](#hardware)). Python 3.14, managed with
> [`uv`](https://docs.astral.sh/uv/).

## The method

Each expert projection weight matrix `W` (shape `n_rows × in_dim`, expert-major)
is quantized as a sum of **codebooks**. Every codebook is fully self-contained:

- **Independent block size.** Codebook `c` splits `in_dim` into
  `n_blocks_c = in_dim // bs_c` sub-vectors of size `bs_c`. Block sizes are
  independent across codebooks and need **not** divide `in_dim`.
- **Per-codebook remainder.** Leftover columns (`in_dim % bs_c`) are stored raw
  (bf16) and reconstructed exactly — so any block size is legal.
- **Primary (c=0): spherical.** Cluster the unit-normalized directions of `W`'s
  blocks (cosine k-means) into a shared codebook of unit centroids, plus a
  per-`(row, block)` scale. `recon₀ = signs₀ ⊙ (scale₀ ⊙ dir₀)`.
- **Residuals (c≥1): real error.** Codebook `c` quantizes the *actual* residual
  `Eᶜ = W − Σ_{c'<c} reconⱼ` over the whole `in_dim` with euclidean k-means
  (magnitude included, no scale of its own). It neither knows nor cares about
  earlier codebooks' block sizes.
- **SSVQ is per-codebook.** With sign-splitting enabled, a codebook folds its
  input to the first orthant (`|·|`), clusters, stores its own signs, and
  reapplies them on reconstruction.
- **Single scale re-fit.** After all codebooks are built, the primary block-wise
  scale is re-fit against the full cumulative reconstruction (least-squares
  optimum given everything else).

Reconstruction is a single shared function (`reconstruct_codebooks`) used by both
the builder (for error/scale-refit) and the decode path, and it streams over
row-chunks so builds fit low-VRAM GPUs.

Storage per codebook: `K_c` centroids × `bs_c` (+ int8 option), plus
`ceil(log2 K_c)` bits/assignment per block, plus the primary's block scales and
optional 1-bit signs, plus the raw remainder.

## Repository layout

```
build.py                 Single-machine driver: build codebooks -> replace experts -> eval ppl
eval_quantized.py        Standalone quantized-model evaluation (CUDA / 3090)
distributed_run.py       Ray-based distributed build
worker.py                TCP worker: reads expert weights straight from safetensors shards
workers.yaml             Cluster inventory (tailscale hosts, per-GPU chunk budgets)

skcq/
  clustering.py          Standalone k-means + build_codebook + reconstruct_codebooks (torch + pt_kmeans only)
  codebook_experts.py    CodebookModule — drop-in quantized replacement for MoE experts (GPU forward)
  quantize.py            extract_and_build_codebooks over all layers
  cuda_server.py         k-means offload server (JSON-RPC over stdin/stdout, tensors via /dev/shm)
  rocm_client.py         client side of the CUDA offload
  orchestrator.py        TCP orchestrator (pickle framing, SSH-launched workers)
  protocol.py, ray_worker.py, metrics.py, ...
  vq/                    Distributed hyperparameter sweep
    server.py            Flask control plane + dashboard
    orchestrator.py      Job queue + SQLite results + worker management
    worker.py            Long-lived sweep worker (loads one layer, pulls jobs)
    runner.py            run_one_kmeans: one config -> {rel_fro_err, bits_per_weight, ...}
    hyperparams.py       Search-space definition + bpw/runtime estimates
    cache.py             On-disk primary-codebook cache
    dashboard/           Static HTML/JS/CSS frontend

configs/                 default.yaml + generated sweep/*.yaml
scripts/                 lint / fmt / typecheck / sweep helpers
experiments/             Ad-hoc analysis (NOT part of the package; see AGENTS.md)
tests/                   test_codebook.py (CPU-only, fast)
some_math/               SEPARATE Lean 4 / mathlib project (quantization theorems)
```

## Environments

This repo is **not** a single environment — pick the venv that matches the task
(details and the ruff/mypy gotcha are in [AGENTS.md](AGENTS.md)):

| venv | torch build | used for |
|------|-------------|----------|
| root `.venv` | (orchestration only) | ray, transformers, the sweep server; `pytest` |
| `rocm/.venv` | ROCm (Strix Halo iGPU) | `build.py`, `distributed_run.py`; **hosts the dev tools** (ruff/mypy/pytest) |
| `cuda/.venv` | CUDA | `eval_quantized.py`, `skcq.cuda_server`, sweep worker on NVIDIA |
| `leopard/.venv` | ROCm (RDNA4 dGPU) | sweep worker on Radeon |

```bash
./setup.sh                 # uv sync in rocm/ and cuda/
(cd leopard && uv sync)    # leopard worker venv
./setup_worker.sh          # remote worker: plain .venv, platform-detected torch
```

## Quickstart

### Build + evaluate one config (single machine)

```bash
# Build codebooks for all layers per configs/default.yaml, replace experts, eval ppl.
# --use-cuda-worker offloads k-means to the CUDA GPU via skcq.cuda_server.
rocm/.venv/bin/python build.py --config configs/default.yaml --compare \
    --use-cuda-worker --baseline-cache baseline.pt

# Evaluate a previously-built quantized model on the 3090:
cuda/.venv/bin/python eval_quantized.py --help
```

### Hyperparameter sweep (distributed, with dashboard)

```bash
# 1. Start the control plane + dashboard on the orchestrator host:
uv run python -m skcq.vq.server --workers workers.yaml \
    --port 5555 --http-port 8050
#    -> open http://localhost:8050/  (Launch / pause / edit search range)

# 2. Start a worker per GPU (locally or SSH'd on remote hosts). Remote workers
#    git pull the repo, so commit + push before launching them:
cuda/.venv/bin/python -m skcq.vq.worker \
    --orchestrator <orch-host>:5555 \
    --model-id Qwen/Qwen3.6-35B-A3B --layer 24 --device cuda --name jaguar-3090
```

Results (`rel_fro_err`, `bits_per_weight`, `compression_ratio`, …) stream into
the orchestrator's SQLite DB and the dashboard. There is also a simpler
single-machine `build.py`-based sweep in `scripts/sweep.sh` /
`scripts/distributed_sweep.sh` (idempotent — skips configs already built).

## Development

```bash
uv run pytest                       # tests/test_codebook.py (CPU-only, fast, no model download)
scripts/lint.sh                     # ruff check + format --check   (needs rocm env)
scripts/fmt.sh                      # ruff check --fix + format      (needs rocm env)
scripts/typecheck.sh                # mypy --follow-imports=skip     (needs rocm env)
```

Lint/typecheck cover only `skcq/` and `build.py`. Style: ruff `line-length=100`,
target py314, `from __future__ import annotations` at the top of every module.
See [AGENTS.md](AGENTS.md) for the important "ruff/mypy live only in the rocm
venv" gotcha.

## Hardware

Tailscale-networked lab (`workers.yaml`): `jaguar` (RTX 3090 CUDA + Strix Halo
ROCm — hosts the k-means offload), `serval` (12 GB NVIDIA), `leopard` (dual
RDNA4 Radeon). `chunk_budget_mb` throttles per-GPU k-means memory. The build and
reconstruction paths are chunked to fit the 12 GB `serval`.

## Notes

- Artifacts are git-ignored: `*.pt` (except `tests/*.pt`), `codebooks*/`,
  `*.safetensors`, `baseline.pt`, `*.log`, `sweep_logs/`.
- `some_math/` is an unrelated Lean 4 / mathlib project.
- Contributor/agent conventions and the multi-venv details live in
  [AGENTS.md](AGENTS.md).
