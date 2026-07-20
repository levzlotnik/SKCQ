#!/usr/bin/env python3
"""Live dashboard for VQ hyperparameter sweep results.

Flask backend serves a single HTML page + a /api/results JSON endpoint.
Frontend is vanilla JS + Plotly (CDN), polls every 5s and re-renders:
  1. Pareto frontier scatter (bpw vs error, color=block_size, size=K)
  2. Heatmap: bs x K -> error (single-codebook)
  3. Best config per bpw bucket (table)

Usage:
    uv run python scripts/vq_dashboard.py
    uv run python scripts/vq_dashboard.py --db vq_results/results.db --port 8050
"""

# ruff: noqa: E501  (inline JS has long lines)
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from flask import Flask, Response, jsonify

REPO = Path(__file__).resolve().parent.parent
DB_PATH = REPO / "vq_results" / "results.db"

app = Flask(__name__)
_DB_PATH: Path = DB_PATH


def _open_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_results() -> list[dict]:
    """Fetch all result rows."""
    conn = _open_db()
    try:
        cur = conn.execute("""
            SELECT projection, scheme, block_size, K, n_codebooks, metric,
                   shared, sign_split, scale_dtype, kmeans_iters,
                   residual_block_sizes, rel_fro_err, bits_per_weight,
                   compression_ratio, worker, completed_at
            FROM results
            ORDER BY bits_per_weight
        """)
        rows = []
        for r in cur.fetchall():
            d = dict(r)
            d["shared"] = bool(d["shared"])
            d["sign_split"] = bool(d["sign_split"])
            try:
                d["residual_block_sizes"] = json.loads(d.get("residual_block_sizes") or "[]")
            except TypeError, json.JSONDecodeError:
                d["residual_block_sizes"] = []
            rows.append(d)
        return rows
    finally:
        conn.close()


def fetch_status() -> dict:
    """Read status.json (written by orchestrator) for sweep progress."""
    status_path = _DB_PATH.parent / "status.json"
    if not status_path.exists():
        return {"running": False, "total": 0, "completed": 0, "failed": 0}
    with open(status_path) as f:
        return json.load(f)


@app.route("/")
def index() -> Response:
    return Response(_HTML, content_type="text/html")


@app.route("/api/results")
def api_results():
    return jsonify({"results": fetch_results(), "status": fetch_status()})


_HTML = """<!DOCTYPE html>
<html>
<head>
  <title>SKCQ VQ Sweep Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body { font-family: -apple-system, system-ui, sans-serif; margin: 20px; background: #fafafa; }
    h1 { margin-bottom: 0.2em; }
    #status { font-size: 14px; color: #666; margin-bottom: 20px; }
    .chart { background: white; border: 1px solid #ddd; padding: 10px; margin-bottom: 20px; }
    table { border-collapse: collapse; width: 100%; font-size: 12px; }
    th, td { border: 1px solid #ddd; padding: 4px 8px; text-align: left; }
    th { background: #f4f4f4; }
    .winner-kmeans { color: #2e7d32; }
    .winner-int { color: #c62828; }
  </style>
</head>
<body>
  <h1>SKCQ VQ Hyperparameter Sweep</h1>
  <div id="status">Loading...</div>

  <div id="pareto" class="chart"></div>
  <div id="heatmap" class="chart"></div>
  <div id="bucket-table" class="chart"></div>

<script>
function isKmeans(r) { return r.scheme && r.scheme.startsWith('kmeans_'); }

function paretoFrontier(rows) {
  // Non-dominated: no other row has bpw <= and err <= (with strict improvement)
  const out = [];
  for (const r of rows) {
    const dominated = rows.some(o =>
      o !== r && o.bits_per_weight <= r.bits_per_weight &&
      o.rel_fro_err <= r.rel_fro_err &&
      (o.bits_per_weight < r.bits_per_weight || o.rel_fro_err < r.rel_fro_err)
    );
    if (!dominated) out.push(r);
  }
  out.sort((a, b) => a.bits_per_weight - b.bits_per_weight);
  return out;
}

function renderPareto(results) {
  const projections = [...new Set(results.map(r => r.projection))];
  const traces = [];
  for (const proj of projections) {
    const projRows = results.filter(r => r.projection === proj);
    const km = projRows.filter(isKmeans);
    const ints = projRows.filter(r => !isKmeans(r));
    if (km.length > 0) {
      traces.push({
        x: km.map(r => r.bits_per_weight),
        y: km.map(r => r.rel_fro_err),
        mode: 'markers',
        name: `${proj} kmeans`,
        text: km.map(r => `${r.scheme}<br>bpw=${r.bits_per_weight.toFixed(3)}<br>err=${r.rel_fro_err.toExponential(3)}`),
        marker: {
          size: km.map(r => Math.max(5, Math.min(15, r.K / 4096 + 4))),
          color: km.map(r => r.block_size),
          colorscale: 'Viridis',
          showscale: true,
          colorbar: {title: 'block_size'},
          opacity: 0.7,
        },
        hovertemplate: '%{text}<extra></extra>',
      });
    }
    if (ints.length > 0) {
      traces.push({
        x: ints.map(r => r.bits_per_weight),
        y: ints.map(r => r.rel_fro_err),
        mode: 'markers',
        name: `${proj} integer`,
        text: ints.map(r => r.scheme),
        marker: {symbol: 'x', size: 8, color: 'red', opacity: 0.6},
        hovertemplate: '%{text}<br>bpw=%{x:.3f}<br>err=%{y:.6f}<extra></extra>',
      });
    }
    const pareto = paretoFrontier(projRows);
    traces.push({
      x: pareto.map(r => r.bits_per_weight),
      y: pareto.map(r => r.rel_fro_err),
      mode: 'lines',
      name: `${proj} Pareto`,
      line: {width: 2, dash: 'dash'},
      hoverinfo: 'skip',
    });
  }
  Plotly.newPlot('pareto', traces, {
    title: 'Pareto Frontier: bits-per-weight vs reconstruction error',
    xaxis: {title: 'bits per weight'},
    yaxis: {title: 'relative Frobenius error'},
    hovermode: 'closest',
    height: 600,
  }, {responsive: true});
}

function renderHeatmap(results) {
  // Single-codebook kmeans only, all projections combined
  const km = results.filter(r => isKmeans(r) && r.n_codebooks === 1);
  if (km.length === 0) {
    document.getElementById('heatmap').innerHTML = '<p>No single-codebook results yet</p>';
    return;
  }
  const bsVals = [...new Set(km.map(r => r.block_size))].sort((a,b) => a-b);
  const kVals = [...new Set(km.map(r => r.K))].sort((a,b) => a-b);
  const errMatrix = bsVals.map(bs => kVals.map(K => {
    const r = km.find(r => r.block_size === bs && r.K === K);
    return r ? r.rel_fro_err : null;
  }));
  const bpwMatrix = bsVals.map(bs => kVals.map(K => {
    const r = km.find(r => r.block_size === bs && r.K === K);
    return r ? r.bits_per_weight.toFixed(2) : '';
  }));
  Plotly.newPlot('heatmap', [{
    z: errMatrix,
    x: kVals.map(String),
    y: bsVals.map(String),
    type: 'heatmap',
    colorscale: 'Viridis_r',
    colorbar: {title: 'error'},
    text: bpwMatrix,
    texttemplate: '%{text}',
    hovertemplate: 'bs=%{y} K=%{x}<br>err=%{z:.6f}<br>bpw=%{text}<extra></extra>',
  }], {
    title: 'Block size × K → error (single codebook, bpw annotated)',
    xaxis: {title: 'K'},
    yaxis: {title: 'block_size'},
    height: 500,
  }, {responsive: true});
}

function renderBucketTable(results) {
  const buckets = [[1.0,1.5],[1.5,2.0],[2.0,2.5],[2.5,3.0],[3.0,3.5],[3.5,4.0],[4.0,5.0],[5.0,6.0]];
  const projections = [...new Set(results.map(r => r.projection))].sort();
  let html = '<h3>Best error per bpw bucket</h3><table><thead><tr><th>bucket</th>';
  for (const p of projections) html += `<th>${p} kmeans</th><th>${p} integer</th><th>winner</th>`;
  html += '</tr></thead><tbody>';
  for (const [lo, hi] of buckets) {
    html += `<tr><td>[${lo.toFixed(1)}-${hi.toFixed(1)})</td>`;
    for (const p of projections) {
      const projRows = results.filter(r => r.projection === p && lo <= r.bits_per_weight && r.bits_per_weight < hi);
      const km = projRows.filter(isKmeans);
      const ints = projRows.filter(r => !isKmeans(r));
      const bestK = km.length > 0 ? Math.min(...km.map(r => r.rel_fro_err)) : null;
      const bestI = ints.length > 0 ? Math.min(...ints.map(r => r.rel_fro_err)) : null;
      html += `<td>${bestK !== null ? bestK.toFixed(6) : '-'}</td>`;
      html += `<td>${bestI !== null ? bestI.toFixed(6) : '-'}</td>`;
      let winner = '-';
      if (bestK !== null && bestI !== null) {
        winner = bestK < bestI ? '<span class="winner-kmeans">kmeans</span>' : '<span class="winner-int">int</span>';
      }
      html += `<td>${winner}</td>`;
    }
    html += '</tr>';
  }
  html += '</tbody></table>';
  document.getElementById('bucket-table').innerHTML = html;
}

async function refresh() {
  try {
    const resp = await fetch('/api/results');
    const data = await resp.json();
    const results = data.results || [];
    const status = data.status || {};
    const progress = status.total > 0 ? (status.completed / status.total * 100).toFixed(1) : 0;
    document.getElementById('status').innerHTML =
      `<strong>Status:</strong> ${status.completed}/${status.total} done ` +
      `(${progress}%) · ${status.failed} failed · running=${status.remaining || 0}`;
    if (results.length > 0) {
      renderPareto(results);
      renderHeatmap(results);
      renderBucketTable(results);
    } else {
      document.getElementById('pareto').innerHTML = '<p>No results yet</p>';
    }
  } catch (e) {
    document.getElementById('status').innerHTML = `<span style="color:red">Error: ${e}</span>`;
  }
}

refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""


def main() -> None:
    global _DB_PATH
    parser = argparse.ArgumentParser(description="VQ sweep dashboard")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    _DB_PATH = args.db
    print(f"Dashboard: http://localhost:{args.port}/  (db={args.db})")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
