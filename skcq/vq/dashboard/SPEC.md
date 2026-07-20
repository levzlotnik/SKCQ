# VQ Sweep Dashboard — Frontend Spec

Three files in this directory: `index.html`, `api.js`, `styles.css`. No build step, no Node, no frameworks. Vanilla JS + Plotly (CDN) + CSS.

## Backend

Flask server at `skcq/vq/server.py`. Base URL = same origin as the HTML.

### GET endpoints

- `GET /api/results` → `{results: [row...], status: {state, total, completed, failed, in_queue, paused}}`
- `GET /api/status` → `{state, total, completed, failed, in_queue, paused}`
- `GET /api/range` → range JSON (see "Range shape" below)
- `GET /api/workers` → array of worker objects (see "Worker shape" below)

### POST endpoints (no body unless noted)

- `POST /api/control/launch` → `{state}`
- `POST /api/control/pause` → `{state}`
- `POST /api/control/resume` → `{state}`
- `POST /api/control/requeue-failed` → `{requeued: N}`
- `POST /api/control/shutdown` → `{state}`
- `POST /api/range` body=range JSON → `{applied: "next_sweep", est_configs: N}`
- `POST /api/workers/<name>/enable` → `{result}`
- `POST /api/workers/<name>/disable` → `{result}`

### Result row shape

```
{projection, scheme, block_size, K, n_codebooks, metric, shared(bool),
 sign_split(bool), scale_dtype, kmeans_iters, residual_block_sizes(array),
 rel_fro_err(float), bits_per_weight(float), compression_ratio(float),
 worker, completed_at}
```

A row is kmeans if `scheme.startsWith("kmeans_")`, else integer baseline.

### Range shape

```
{
  projection: ["gate", "down"],
  bpw_min: 1.0,
  bpw_max: 6.0,
  primary: {block_size: [8,10,12,16,24,32,64,128], K: [16,32,...,1048576],
            metric: ["cosine"], sign_split: [true], scale_dtype: ["int8"]},
  residuals: [{block_size: [8,16,32], K: [16,...,8192]}, ...]
}
```

`primary.metric`, `primary.sign_split`, `primary.scale_dtype` are absent on residual codebook entries (residuals are always euclidean / no sign split / fp16).

### Worker shape

```
{
  name: "jaguar-3090",
  host: "localhost",
  enabled: true,
  connected: true,
  devices: [{index: 0, name: "NVIDIA RTX 3090", total_vram_mb: 24576}],
  current_job: "gate_p_b8_K256_mcos_ss1_sdint8" | null,
  last_heartbeat: {t: 1784511332.0, devices: [{idx:0, alloc_mb:512, reserved_mb:1024, used_mb:2048, util_pct:47.5}]},
  history: [/* up to 60 heartbeats, same shape as last_heartbeat */]
}
```

## UI Layout

Four sections stacked vertically.

### 1. Status bar (top)

Text: `State: <state> | Progress: <completed>/<total> (<pct>%) | Failed: <failed> | In queue: <in_queue>`

Buttons: **Launch**, **Pause**, **Resume**, **Requeue failed**, **Shutdown**.

Button enable rules:
- idle/stopped: Launch=on, Pause/Resume/Shutdown=off
- running: Pause=on, Shutdown=on, Launch/Resume=off
- paused: Resume=on, Shutdown=on, Launch/Pause=off

### 2. Hyperparameter range panel

**Target BPW range**: dual-handle slider [1.0, 6.0] step 0.1. Show `[min, max]` text.

**Primary codebook**:
- Metric: radio `euclidean` / `cosine`
- Sign split: radio `yes` / `no`
- Block size: checkboxes [8, 10, 12, 16, 24, 32, 64, 128]
- K: checkboxes [16, 32, 64, 128, 256, 512, 1K, 2K, 4K, 8K, 16K, 32K, 64K, 128K, 256K, 512K, 1M] (display human-readable, value is raw int)
- If metric=cosine: Scale dtype — two options side by side:
  - Radio `int` + dual-handle slider for bits (2 to 16, step 1)
  - Radio `fp` + checkboxes [fp8_e5m2, fp8_e4m3, fp16, bf16, fp32]

**Residual codebooks** (dynamic list):
- Each slot: Block size checkboxes [8,10,12,16,24,32,64,128], K checkboxes (same as primary), `[×]` remove button
- `[+] Add residual codebook` button
- Residuals are always euclidean/no-sign/fp16 — don't show those controls

**Apply button**: POST current range to `/api/range`. Show banner: "Changes will apply to next sweep".

**Live estimate** next to Apply: `~<est_configs> configs | ~<est_runtime>h` (est_configs from POST response, est_runtime ≈ est_configs × 30s / worker_count).

### 3. Results charts

**a) Pareto scatter**: x=bits_per_weight, y=rel_fro_err. Kmeans = circles (color=block_size Viridis, size ~ K). Integers = red ×. Pareto frontier = dashed line (compute non-dominated in JS: a point is dominated if another has bpw ≤ and err ≤ with one strict). One trace set per projection.

**b) Heatmap**: block_size (y) × K (x) → rel_fro_err (color), single-codebook kmeans only (n_codebooks==1). Annotate cells with bpw. Dropdown to select projection.

**c) Best-per-bpw table**: buckets [1.0-1.5), [1.5-2.0), ..., [5.0-6.0). Columns per projection: best kmeans err, best int err, winner (green=kmeans, red=int).

### 4. GPU cards

One card per worker. Each shows:
- Name (e.g., "Jaguar 3090")
- Status dot: green=connected, gray=disconnected
- Enable/Disable toggle (POST to corresponding endpoint)
- Device name + total VRAM
- VRAM bar: `████████░░ 8.2 / 24.0 GB` (used_mb / total_vram_mb from last heartbeat)
- Utilization % (util_pct from last heartbeat)
- Plotly time-series: x=time (heartbeat `t`), two lines: VRAM used (MB) and utilization (%). Data from `history` ring buffer.
- Current job name if set

## Polling

- `/api/status` every 2s → status bar + button states
- `/api/results` every 5s → all 3 result charts
- `/api/workers` every 5s → GPU cards + GPU charts
- `/api/range` on page load + after Apply

## Constraints

- Plotly from CDN: `https://cdn.plot.ly/plotly-2.35.2.min.js`
- No other JS libraries
- CSS in `styles.css`, JS in `api.js`
- K values display: format integers ≥ 1024 as `N K` / `N M` (e.g., 65536 → "64K", 1048576 → "1M")
- `residual_block_sizes` is already a parsed array in the JSON response
- `shared` and `sign_split` are booleans
- Heartbeat `t` is Unix timestamp float seconds

## Server-side changes needed

Update `skcq/vq/server.py`:
- `DASHBOARD_HTML` constant → `DASHBOARD_DIR = Path(__file__).resolve().parent / "dashboard"`
- `/` route → `send_file(DASHBOARD_DIR / "index.html")`
- Add `@app.route('/styles.css')` → `send_file(DASHBOARD_DIR / 'styles.css', content_type='text/css')`
- Add `@app.route('/api.js')` → `send_file(DASHBOARD_DIR / 'api.js', content_type='application/javascript')`

(Do this in the same PR/commit as the frontend files.)
