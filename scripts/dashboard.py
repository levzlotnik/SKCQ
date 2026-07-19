"""Generate a Plotly HTML dashboard from sweep results.

Reads sweep_results/unified.csv (from collect_sweep.py) and writes
sweep_results/dashboard.html with:
  1. Pareto frontier scatter (bpw vs error, colored by block_size)
  2. Heatmap: bs x K -> error (single-codebook)
  3. Best config per bpw bucket table
  4. Integer baselines overlay

Usage:
    uv run python scripts/dashboard.py
    uv run python scripts/dashboard.py --input sweep_results/unified.csv \
        --out sweep_results/dashboard.html
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "sweep_results"


def load_rows(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                row["block_size"] = int(row["block_size"]) if row["block_size"] else 0
                row["K"] = int(row["K"]) if row["K"] else 0
                row["n_codebooks"] = int(row["n_codebooks"]) if row["n_codebooks"] else 0
                row["rel_fro_err"] = float(row["rel_fro_err"])
                row["bits_per_weight"] = float(row["bits_per_weight"])
                row["compression_ratio"] = float(row["compression_ratio"])
            except (ValueError, KeyError):
                continue
            rows.append(row)
    return rows


def is_kmeans(row: dict) -> bool:
    return row["scheme"].startswith("kmeans_")


def pareto_frontier(rows: list[dict]) -> list[dict]:
    """Compute non-dominated frontier (minimize bpw and err)."""
    pareto = []
    for r in rows:
        bpw, err = r["bits_per_weight"], r["rel_fro_err"]
        dominated = False
        for o in rows:
            if o is r:
                continue
            if (
                o["bits_per_weight"] <= bpw
                and o["rel_fro_err"] <= err
                and (o["bits_per_weight"] < bpw or o["rel_fro_err"] < err)
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(r)
    pareto.sort(key=lambda r: r["bits_per_weight"])
    return pareto


def fig_pareto(rows: list[dict]) -> go.Figure:
    """Pareto scatter: bpw vs error, colored by block_size, size by K."""
    fig = go.Figure()

    for proj in sorted({r["projection"] for r in rows}):
        proj_rows = [r for r in rows if r["projection"] == proj]
        km = [r for r in proj_rows if is_kmeans(r)]
        ints = [r for r in proj_rows if not is_kmeans(r)]

        # Kmeans points
        fig.add_trace(
            go.Scatter(
                x=[r["bits_per_weight"] for r in km],
                y=[r["rel_fro_err"] for r in km],
                mode="markers",
                name=f"{proj} kmeans",
                text=[r["scheme"] for r in km],
                marker=dict(
                    size=[max(5, min(15, r["K"] / 4096 + 4)) for r in km],
                    color=[r["block_size"] for r in km],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="block_size"),
                    opacity=0.7,
                ),
                hovertemplate="%{text}<br>bpw=%{x:.3f}<br>err=%{y:.6f}<extra></extra>",
            )
        )

        # Integer baselines
        fig.add_trace(
            go.Scatter(
                x=[r["bits_per_weight"] for r in ints],
                y=[r["rel_fro_err"] for r in ints],
                mode="markers",
                name=f"{proj} integer",
                text=[r["scheme"] for r in ints],
                marker=dict(symbol="x", size=8, color="red", opacity=0.6),
                hovertemplate="%{text}<br>bpw=%{x:.3f}<br>err=%{y:.6f}<extra></extra>",
            )
        )

        # Pareto frontier line (kmeans + int combined)
        pareto = pareto_frontier(proj_rows)
        fig.add_trace(
            go.Scatter(
                x=[r["bits_per_weight"] for r in pareto],
                y=[r["rel_fro_err"] for r in pareto],
                mode="lines",
                name=f"{proj} Pareto",
                line=dict(width=2, dash="dash"),
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title="Pareto Frontier: bits-per-weight vs reconstruction error",
        xaxis_title="bits per weight",
        yaxis_title="relative Frobenius error",
        hovermode="closest",
        height=700,
    )
    return fig


def fig_heatmap_bs_k(rows: list[dict], projection: str) -> go.Figure:
    """Heatmap: block_size x K -> error, for single-codebook kmeans."""
    km = [
        r for r in rows if is_kmeans(r) and r["projection"] == projection and r["n_codebooks"] == 1
    ]
    if not km:
        fig = go.Figure()
        fig.update_layout(title=f"No single-codebook data for {projection}")
        return fig

    bs_vals = sorted({r["block_size"] for r in km})
    k_vals = sorted({r["K"] for r in km})

    # Build error matrix
    err_matrix = [[None] * len(k_vals) for _ in range(len(bs_vals))]
    bpw_matrix = [[None] * len(k_vals) for _ in range(len(bs_vals))]
    for r in km:
        i = bs_vals.index(r["block_size"])
        j = k_vals.index(r["K"])
        err_matrix[i][j] = r["rel_fro_err"]
        bpw_matrix[i][j] = r["bits_per_weight"]

    fig = go.Figure(
        data=go.Heatmap(
            z=err_matrix,
            x=[str(k) for k in k_vals],
            y=[str(bs) for bs in bs_vals],
            colorscale="Viridis_r",
            colorbar=dict(title="error"),
            text=bpw_matrix,
            texttemplate="%{text:.2f}",
            hovertemplate="bs=%{y} K=%{x}<br>err=%{z:.6f}<br>bpw=%{text:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{projection}: block_size x K -> error (single codebook, bpw annotated)",
        xaxis_title="K",
        yaxis_title="block_size",
        height=500,
    )
    return fig


def fig_best_per_bpw(rows: list[dict]) -> go.Figure:
    """Bar chart: best error per bpw bucket, kmeans vs integer."""
    buckets = [
        (1.0, 1.5),
        (1.5, 2.0),
        (2.0, 2.5),
        (2.5, 3.0),
        (3.0, 3.5),
        (3.5, 4.0),
        (4.0, 5.0),
        (5.0, 6.0),
    ]
    bucket_labels = [f"[{lo:.1f}-{hi:.1f})" for lo, hi in buckets]

    fig = make_subplots(rows=1, cols=1)
    for proj in sorted({r["projection"] for r in rows}):
        proj_rows = [r for r in rows if r["projection"] == proj]
        km = [r for r in proj_rows if is_kmeans(r)]
        ints = [r for r in proj_rows if not is_kmeans(r)]

        km_best = []
        int_best = []
        for lo, hi in buckets:
            bk = [r for r in km if lo <= r["bits_per_weight"] < hi]
            bi = [r for r in ints if lo <= r["bits_per_weight"] < hi]
            km_best.append(min((r["rel_fro_err"] for r in bk), default=None))
            int_best.append(min((r["rel_fro_err"] for r in bi), default=None))

        fig.add_trace(
            go.Bar(
                x=bucket_labels,
                y=[v if v is not None else 0 for v in km_best],
                name=f"{proj} kmeans",
                opacity=0.7,
            )
        )
        fig.add_trace(
            go.Bar(
                x=bucket_labels,
                y=[v if v is not None else 0 for v in int_best],
                name=f"{proj} integer",
                opacity=0.7,
            )
        )

    fig.update_layout(
        title="Best error per bpw bucket: kmeans vs integer",
        xaxis_title="bpw bucket",
        yaxis_title="best relative Frobenius error",
        barmode="group",
        height=500,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sweep dashboard")
    parser.add_argument("--input", type=Path, default=RESULTS_DIR / "unified.csv")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "dashboard.html")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input {args.input} not found. Run collect_sweep.py first.")
        return

    rows = load_rows(args.input)
    print(f"Loaded {len(rows)} rows from {args.input}")

    # Build figures
    figs = []
    figs.append(("Pareto Frontier", fig_pareto(rows)))
    for proj in sorted({r["projection"] for r in rows}):
        figs.append((f"Heatmap: {proj} bs x K", fig_heatmap_bs_k(rows, proj)))
    figs.append(("Best per bpw", fig_best_per_bpw(rows)))

    # Combine into a single HTML
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("<html><head><title>SKCQ Sweep Dashboard</title>")
        f.write("<style>body{font-family:sans-serif;margin:20px} h2{margin-top:40px}</style>")
        f.write("</head><body><h1>SKCQ Sweep Dashboard</h1>")
        for title, fig in figs:
            f.write(f"<h2>{title}</h2>")
            f.write(fig.to_html(full_html=False, include_plotlyjs="cdn"))
        f.write("</body></html>")

    print(f"Wrote {args.out}")
    print(f"Open with: file://{args.out.resolve()}")


if __name__ == "__main__":
    main()
