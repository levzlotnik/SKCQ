# AGENTS.md

SKCQ: post-training quantization of MoE experts (Qwen3.6-35B-A3B) via spherical
product-quantization codebooks + residuals. Python 3.14, uv-managed.

## Three separate uv projects / venvs (most important thing to get right)

This repo is NOT a single environment. Pick the venv that matches the task:

- **root `.venv`** (`pyproject.toml`): orchestration deps (ray, transformers).
  Dev tools (ruff/mypy/pytest) are NOT here — see gotcha below.
- **`rocm/.venv`** (`rocm/pyproject.toml`): ROCm torch for the Strix Halo iGPU
  (gfx1151). Runs `build.py` and `distributed_run.py`. Also declares the dev
  tools in its `[dependency-groups] dev` (ruff, mypy, pytest).
- **`cuda/.venv`** (`cuda/pyproject.toml`): CUDA torch for the RTX 3090. Runs
  `eval_quantized.py` and the k-means offload server (`skcq.cuda_server`).
- **`leopard/.venv`** (`leopard/pyproject.toml`): ROCm torch for RDNA4 dGPUs
  (gfx1201, e.g. Radeon RX 9700). Sweep worker only — runs
  `experiments/weight_quant_error.py`. Wheel index: `gfx120X-all/`.

Create both GPU venvs with `./setup.sh` (runs `uv sync` in `rocm/` and `cuda/`).
Create the leopard venv with `(cd leopard && uv sync)`.
Remote workers use `./setup_worker.sh` which creates a plain `.venv` with a
platform-detected torch build.

**Gotcha:** `ruff` and `mypy` live ONLY in the `rocm` project's dev group, so
bare `uv run ruff`/`uv run mypy` from the repo root fails (`Failed to spawn:
ruff`) — and `scripts/lint.sh`/`scripts/typecheck.sh` call exactly that. Run them
from the rocm environment: `uv run --project rocm ruff ...` / activate
`rocm/.venv`, or after `setup.sh`. `pytest` is NOT affected — it's present in the
root env transitively, so `uv run pytest` works from root as-is.

## Commands

`uv run` is the standard command prefix in this repo (scripts use it throughout;
GPU drivers are instead invoked with an explicit venv python, e.g.
`rocm/.venv/bin/python build.py`, `cuda/.venv/bin/python eval_quantized.py`).

- Lint: `scripts/lint.sh` — `ruff check` + `ruff format --check` (needs rocm env; see gotcha)
- Format: `scripts/fmt.sh` — `ruff check --fix` + `ruff format` (needs rocm env)
- Typecheck: `scripts/typecheck.sh` — `mypy --follow-imports=skip` (needs rocm env)
- Tests: `uv run pytest` (works from root; config in root `pyproject.toml`,
  `pythonpath=["."]`). Only `tests/test_codebook.py` — CPU-only, fast, no
  GPU/model download needed.
- Single test: `uv run pytest tests/test_codebook.py::TestBuildCodebook::test_output_shapes`

Lint/typecheck cover **only `skcq/` and `build.py`**. The other root scripts
(`distributed_run.py`, `worker.py`, `eval_quantized.py`) and `experiments/`,
`poc/` are excluded — don't assume they're checked.

Style: ruff `line-length = 100`, target py314; `from __future__ import annotations`
at the top of every module.

## Architecture

- `build.py` — single-machine driver: build codebooks, replace experts, eval ppl.
  Runs on ROCm. `--use-cuda-worker` offloads `build_codebook` (k-means) to a CUDA
  subprocess (`skcq/rocm_client.py` -> `skcq/cuda_server.py`, JSON-RPC over
  stdin/stdout, tensors via `/dev/shm`). This is why jaguar needs both venvs.
- `skcq/clustering.py` — standalone k-means + `build_codebook` +
  `reconstruct_codebooks` (only torch + pt_kmeans); imported by both ROCm and
  CUDA sides. `CodebookParams`/`LayerOverride`/`CodebookResult` live here.
- `skcq/codebook_experts.py` — `CodebookModule` (PQ + additive residual codebooks),
  drop-in replacement for MoE experts. Assignments/signs are buffers, codebooks/
  scales are Parameters. Each codebook carries its own block size, remainder, and
  optional signs (see Codebook scheme).
- **Two coexisting distributed paths** (don't conflate):
  - Ray: `distributed_run.py` + `skcq/ray_worker.py` (`ray.init(address="auto")`).
  - TCP: `skcq/orchestrator.py` + `worker.py` (length-prefixed pickle over TCP,
    SSH-launched remote workers, `skcq/protocol.py`). `worker.py` reads expert
    weights directly from safetensors shards (no transformers/model load).
- **VQ hyperparameter sweep** (`skcq/vq/`) — a separate distributed system from
  the two above: `skcq.vq.server` (Flask control plane + dashboard) +
  `skcq.vq.orchestrator` (job queue + SQLite results) + `skcq.vq.worker`
  (long-lived, loads one layer, pulls configs, calls `runner.run_one_kmeans`).
  Search space + bpw/runtime estimates in `skcq/vq/hyperparams.py`;
  primary-codebook cache in `skcq/vq/cache.py`.
- Workflow is two-phase: build codebooks (ROCm/distributed) THEN quantized eval
  separately on the 3090 (`eval_quantized.py`). `baseline.pt` caches baseline
  ppl/routing/KLD and is auto-invalidated when `--kld-tokens` changes.

### Codebook scheme (`build_codebook` / `reconstruct_codebooks`)

Each codebook is self-contained; do NOT assume a shared block grid:
- **Independent block size per codebook.** `bs_c` need not divide `in_dim`;
  `n_blocks_c = in_dim // bs_c` and the leftover `in_dim % bs_c` columns are a
  **raw bf16 remainder** stored per codebook and reconstructed exactly.
- **Primary (c=0)** is spherical (cosine): unit-direction centroids + a
  per-`(row, block)` scale. **Residuals (c≥1)** quantize the REAL error
  `W − Σ_{c'<c} recon` with euclidean k-means (magnitude included, no own scale).
- **SSVQ is per-codebook**: `CodebookResult.sign_bits` is a list (one entry per
  codebook, `(n_rows, cov_c)` or None), applied to that codebook's contribution.
- One **final scale re-fit** of the primary block scale against the full
  reconstruction. `reconstruct_codebooks` is the single source of truth (used by
  both `build_codebook` and `CodebookResult.reconstruct`) and **streams over
  row-chunks** so builds fit low-VRAM GPUs (12 GB `serval`).
- Runner/hyperparams bpw accounting must stay identical; both count per-codebook
  remainder bits and per-codebook sign bits. There is NO block-size divisibility
  filter — any block size is valid.

## Hardware / workers (`workers.yaml`)

Tailscale hosts: `jaguar` (orchestrator + 3090 CUDA + Strix ROCm), `serval`
(12GB VRAM, small `chunk_budget_mb`), `tiger` (commented out). Remote workers
`git pull` and run over SSH. `chunk_budget_mb` throttles k-means memory per GPU.

## Sweeps

`scripts/gen_sweep_configs.sh` generates `configs/sweep/*.yaml` (grid over K/block
ratio, n_blocks, n_codebooks). `scripts/sweep.sh` (local ROCm+CUDA offload) and
`scripts/distributed_sweep.sh <config> [workers.yaml]` run them; results land in
`codebooks_*/compare*.json`, logs in `sweep_logs/`. Both are idempotent — they
skip layers/configs whose codebooks already exist.

## Experiments (`experiments/`)

Ad-hoc research/analysis scripts. NOT part of the `skcq` package and NOT
linted/typechecked/tested (excluded from `scripts/*`). Most `import skcq` via a
`sys.path.insert(..)` hack and download the real Qwen 35B model to read layer-24
gate/up/down weights — run them with a torch-capable venv (`rocm/.venv` or
`cuda/.venv`), not the root `.venv`.

- **`weight_quant_error.py`** — the maintained analysis. Compares integer
  baselines (int8/int4/fp8, per-tensor + per-channel) vs spherical k-means
  codebooks; reports relative Frobenius error, bits/weight, compression ratio.
  Real argparse CLI (`--block-size` and `--K` required; `--layer` default 24;
  `--sign-split` = SSVQ, `--shared`, `--metric`, `--n-codebooks`, `--krm`,
  `--kmeans-iters`, `--output`). Writes a CSV and prints a table.
- **`zador_estimate.py`** — pure-math Zador rate–distortion estimate (no torch,
  no model); standalone, just prints a table.
- **`test_*.py`** (`test_kmeans`, `test_recon`, `test_build_codebook`,
  `test_cluster_block`, `test_large_block`, `test_reshape`, `test_reshape2`) —
  one-off debugging scripts that print output; **not pytest tests** despite the
  prefix (no assertions/fixtures). pytest only collects `tests/`. `test_kmeans`
  and `test_recon` use synthetic data; the rest download the 35B model. Run
  directly (`rocm/.venv/bin/python experiments/test_recon.py`), never via pytest.
- **`weight_quant_error.csv` / `weight_quant_error_l24.csv`** — committed output
  tables from `weight_quant_error.py` (unlike other `*.csv`/artifacts, these are
  tracked).

## Artifacts — do not commit

`.gitignore` excludes `*.pt` (except `tests/*.pt`), `codebooks*/`, `*.safetensors`,
`baseline.pt`, `*.log`, `sweep_logs/`, `poc/`.

## opencode config

- `opencode.json` wires the `lean-lsp` MCP (`LEAN_PROJECT_PATH=some_math`) and an
  `arxiv` MCP. Agent templates live in `.opencode/agents/`; `./make_agent.sh
  <provider/model> <implementation|exploration|lean4>` generates a concrete agent.
  Agent permission rules gate build/sweep commands (`ask`) and deny destructive git.
- `some_math/` is a **separate Lean 4 / mathlib project** (quantization theorems).
  Use the `lean4` skill and Lean agents for it — it is unrelated to the Python code.
