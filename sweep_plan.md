# Sweep & Optimize Plan

## Goal

Find the Pareto-optimal VQ quantization configuration for every bpw level (1.0–6.0),
across the full search space of block sizes, K values, residual hierarchies, and
codebook/scale quantization. Run distributed across all machines, collect into a
unified dataset, visualize, and refine with Bayesian optimization.

## Architecture

### 1. Config Generator (`scripts/gen_sweep.py`)

Generates all valid experiment configurations as a list of CLI argument strings.

**Search space:**

| Parameter | Values | Notes |
|-----------|--------|-------|
| `--block-size` | 8, 9, 10, 11, 12, 14, 16, 20, 24, 32, 48, 64, 128 | Must divide 2048 (skip otherwise) |
| `--K` | 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072 | K ≤ n_rows (131072 for gate) |
| `--n-codebooks` | 1, 2, 3 | |
| `--residual-k` | None, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 | Only when n_codebooks ≥ 2 |
| `--residual-block-sizes` | None, 8, 16, 32 | Must divide or be multiple of block_size |
| `--codebook-bits` | 8, 16 | |
| `--scale-dtype` | int8 | Fixed for now |
| `--metric` | cosine | Fixed (SSVQ + spherical always wins) |
| `--shared` | True | Fixed (always shared) |
| `--sign-split` | True | Fixed (always SSVQ) |
| `--kmeans-iters` | 50 | Fixed for sweep; higher for final runs |

**Filtering rules (applied before distribution):**
1. `2048 % block_size` → skip if nonzero remainder > 6 (tiny remainder ok)
2. `residual_block_sizes` must divide or be multiple of `block_size`
3. Estimated bpw must be in [1.0, 6.0]
4. Estimated runtime > 10min → skip (K too large relative to n_rows)
5. Duplicate configs (same bs/K/rbs/rK after normalization) → skip

**Estimated grid size:** ~500-800 configs after filtering.

**Output:** JSON file `sweep_configs.json` — list of `{id, args, est_bpw, est_runtime}`.

### 2. Distributed Runner (`scripts/run_sweep_distributed.py`)

Reads `sweep_configs.json`, distributes across workers, runs in parallel.

**Workers (from `workers.yaml`):**
- `jaguar-3090`: local CUDA (RTX 3090, 24GB) — 4 parallel workers
- `jaguar-strix`: local ROCm (Strix Halo iGPU) — 1 worker (slow, for small K)
- `serval`: remote CUDA (12GB VRAM) — 2 parallel workers (via SSH)
- `tiger`: remote CPU (M3 Ultra, 512GB UMA) — 1 worker (for CPU-compatible configs)

**Distribution strategy:**
- Sort configs by estimated runtime (descending)
- Assign to workers round-robin (longest-first = load-balanced)
- Each worker runs: `uv run python experiments/weight_quant_error.py <args> --output sweep_results/<id>.csv`
- Workers write individual CSVs (no shared file → no contention)
- Heartbeat: workers report progress to a local status file every 30s
- Idempotent: skip configs whose output CSV already exists

**Failure handling:**
- Retry once on timeout/OOM
- Log failures to `sweep_results/failed.jsonl`
- Continue on failure (don't block other configs)

**Local parallelism:**
- jaguar-3090: 4 processes (GPU has 24GB, each ~4-5GB for K=65k)
- Use `xargs -P 4` or Python `multiprocessing.Pool(4)`

**Remote workers (serval/tiger):**
- SSH launch: `ssh serval "cd ~/Projects/SKCQ && git pull && uv run python ..."`
- Each remote worker runs its assigned configs sequentially
- Results collected via `scp` after completion, or shared via Tailscale filesystem

### 3. Result Collector (`scripts/collect_sweep.py`)

After all workers finish (or periodically during the sweep):

1. Scan `sweep_results/*.csv` for completed experiments
2. Merge into `sweep_results/unified.csv` (deduplicated, sorted by bpw)
3. Compute Pareto frontier (non-dominated configs by bpw and error)
4. Print summary table: top-5 configs per bpw bucket (1.0-1.5, 1.5-2.0, 2.0-2.5, ...)

### 4. Dashboard (`scripts/dashboard.py` → `sweep_results/dashboard.html`)

Static HTML (no server needed) using Plotly for offline rendering.

**Visualizations:**

1. **Pareto frontier scatter** (bpw vs error)
   - Color by block_size, size by K, shape by n_codebooks
   - Highlight Pareto-optimal configs (bold markers)
   - Overlay integer baselines (int2/3/4/8 per-block, per-channel)
   - Interactive hover: full config label, bpw, error, CR

2. **Heatmap: bs × K → error** (for single-codebook)
   - X-axis: K (log scale)
   - Y-axis: block_size
   - Color: error (with bpw annotation)
   - Separate heatmaps for cb=1, cb=2, cb=3

3. **Heatmap: rbs × rK → error** (for cb=2, fixed primary)
   - X-axis: residual_k
   - Y-axis: residual_block_size
   - Color: error
   - Shows the `log2(K_r)/bs_r` invariant visually

4. **Best config per bpw** (table)
   - Bucket: [1.0-1.5), [1.5-2.0), [2.0-2.5), [2.5-3.0), [3.0-3.5), [3.5-4.0), [4.0-5.0), [5.0-6.0]
   - Columns: bpw, error, CR, config label, vs best integer at that bpw

5. **Zador ratio heatmap** (bs × K → real_error / zador_error)
   - Shows where k-means is most/least efficient
   - Helps identify regions worth deeper exploration

### 5. Bayesian Optimization (`scripts/bayesopt.py`)

After the grid sweep, refine around promising regions.

**Framework:** Optuna (TPESampler, no external server needed)

**Objective:** minimize `rel_fro_err` subject to `bpw ∈ [target_low, target_high]`

**Search space (continuous, then rounded to valid values):**
```python
def suggest(trial):
    bs = trial.suggest_int("block_size", 6, 128)
    K = trial.suggest_categorical("K", [2**i for i in range(8, 18)])
    n_cb = trial.suggest_int("n_codebooks", 1, 3)
    if n_cb >= 2:
        rbs = trial.suggest_int("residual_block_size", 4, 64)
        rK = trial.suggest_categorical("residual_k", [2**i for i in range(4, 14)])
    else:
        rbs = None
        rK = None
    codebook_bits = trial.suggest_categorical("codebook_bits", [8, 16])
    return {bs, K, n_cb, rbs, rK, codebook_bits}
```

**Constraints:**
- bs must divide 2048 (skip invalid, return Inf)
- rbs must divide or be multiple of bs
- bpw must be in target range (soft constraint via penalty)

**Seeding:**
- Pre-populate with all grid sweep results (warm start)
- TPE sampler uses these to focus on promising regions

**Trials:**
- 200-500 trials per bpw target
- Early stopping: if error > 2× current best at same bpw, prune
- Parallel: Optuna supports distributed trials across workers

**Output:**
- `sweep_results/bayesopt_<target>.csv` with all trials
- Update dashboard with BO-found configs

## Execution Plan (Tomorrow)

### Morning: Infrastructure (~2h)

1. [ ] Write `scripts/gen_sweep.py` — config generator + filtering
2. [ ] Write `scripts/run_sweep_distributed.py` — distributed runner
3. [ ] Write `scripts/collect_sweep.py` — result collector + Pareto
4. [ ] Test on 5-10 configs locally (jaguar-3090 only)

### Midday: Launch Sweep (~30min setup, ~4-8h runtime)

5. [ ] `git pull` on serval/tiger
6. [ ] Launch full sweep across all workers
7. [ ] Monitor progress via status file + `tail -f sweep_logs/*.log`

### Afternoon: Dashboard + Analysis (~2h)

8. [ ] Write `scripts/dashboard.py` — Plotly HTML
9. [ ] Collect results, generate dashboard
10. [ ] Analyze Pareto frontier, identify promising regions
11. [ ] Identify top-10 configs per bpw bucket

### Evening: Bayesian Optimization (~2h setup, overnight run)

12. [ ] Write `scripts/bayesopt.py` — Optuna sweep
13. [ ] Seed with grid results, launch BO overnight
14. [ ] Target bpw regions: [1.5-2.0], [2.0-2.5], [3.0-4.0]

### Next Day: Final Results

15. [ ] Collect BO results, update dashboard
16. [ ] Run top-5 configs with 300 iterations (high-quality final run)
17. [ ] Down/up projection experiments (in_dim=512, different distribution)
18. [ ] Write up findings

## File Layout

```
scripts/
  gen_sweep.py              # config generator
  run_sweep_distributed.py  # distributed runner
  collect_sweep.py          # result collector + Pareto
  dashboard.py              # Plotly HTML dashboard
  bayesopt.py               # Optuna Bayesian optimization

sweep_results/
  unified.csv               # all results merged
  pareto.csv                # Pareto-optimal only
  dashboard.html            # visualization
  bayesopt_*.csv            # BO trial results
  status.json               # sweep progress (updated every 30s)
  failed.jsonl              # failed configs

sweep_logs/
  <worker_name>_<config_id>.log  # per-config logs
```

## Key Constraints

- Each experiment takes 30s-5min (K=65k at bs=8 is longest)
- GPU memory: K=131072 × bs=128 × fp32 ≈ 64MB centroids → fine for 24GB 3090
- serval (12GB): limit K ≤ 32768 for bs ≤ 32
- tiger (CPU): 4-10× slower per config, assign small-K only
- Total estimated sweep time: 500 configs × avg 2min / 7 parallel workers ≈ 2.5h
