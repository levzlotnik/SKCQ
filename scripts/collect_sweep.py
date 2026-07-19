"""Collect per-config sweep CSVs into a unified dataset, compute Pareto frontier.

Usage:
    uv run python scripts/collect_sweep.py
    uv run python scripts/collect_sweep.py --out sweep_results/unified.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "sweep_results"

FIELDNAMES = [
    "projection",
    "scheme",
    "block_size",
    "K",
    "n_codebooks",
    "metric",
    "shared",
    "sign_split",
    "scale_dtype",
    "kmeans_iters",
    "residual_block_sizes",
    "rel_fro_err",
    "bits_per_weight",
    "compression_ratio",
]


def is_pareto_dominated(row: dict, others: list[dict]) -> bool:
    """A row is dominated if some other row has bpw <= and err <= (and strictly better on one)."""
    bpw = float(row["bits_per_weight"])
    err = float(row["rel_fro_err"])
    for o in others:
        o_bpw = float(o["bits_per_weight"])
        o_err = float(o["rel_fro_err"])
        if o_bpw <= bpw and o_err <= err and (o_bpw < bpw or o_err < err):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect sweep results")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "unified.csv")
    parser.add_argument("--pareto-out", type=Path, default=RESULTS_DIR / "pareto.csv")
    args = parser.parse_args()

    # Collect all per-config CSVs
    csv_files = sorted(args.results_dir.glob("*.csv"))
    # Exclude unified.csv / pareto.csv themselves
    csv_files = [f for f in csv_files if f.name not in ("unified.csv", "pareto.csv")]
    print(f"Found {len(csv_files)} per-config CSVs")

    all_rows: list[dict] = []
    for f in csv_files:
        with open(f) as cf:
            reader = csv.DictReader(cf)
            for row in reader:
                # Parse numeric fields
                try:
                    row["block_size"] = int(row["block_size"])
                    row["K"] = int(row["K"])
                    row["n_codebooks"] = int(row["n_codebooks"])
                    row["shared"] = row["shared"] == "True"
                    row["sign_split"] = row["sign_split"] == "True"
                    row["kmeans_iters"] = int(row["kmeans_iters"])
                    row["rel_fro_err"] = float(row["rel_fro_err"])
                    row["bits_per_weight"] = float(row["bits_per_weight"])
                    row["compression_ratio"] = float(row["compression_ratio"])
                except (ValueError, KeyError):
                    continue
                row["_source_file"] = f.name
                all_rows.append(row)

    print(f"Total rows: {len(all_rows)}")

    # Separate kmeans from integer baselines
    kmeans_rows = [r for r in all_rows if r["scheme"].startswith("kmeans_")]
    int_rows = [r for r in all_rows if not r["scheme"].startswith("kmeans_")]
    print(f"  kmeans: {len(kmeans_rows)}, integer baselines: {len(int_rows)}")

    # Dedupe integer baselines (same projection + scheme appears in every config CSV)
    seen_int = set()
    int_dedup = []
    for r in int_rows:
        key = (r["projection"], r["scheme"], r["block_size"])
        if key in seen_int:
            continue
        seen_int.add(key)
        int_dedup.append(r)
    print(f"  integer baselines (deduped): {len(int_dedup)}")

    # Write unified CSV (kmeans + deduped integer baselines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for r in kmeans_rows + int_dedup:
            writer.writerow({k: r.get(k, "") for k in FIELDNAMES})
    print(f"Wrote {args.out} ({len(kmeans_rows) + len(int_dedup)} rows)")

    # Per-projection Pareto frontier (kmeans only, then with int baselines)
    projections = sorted({r["projection"] for r in all_rows})
    pareto_rows = []
    for proj in projections:
        proj_kmeans = [r for r in kmeans_rows if r["projection"] == proj]
        proj_int = [r for r in int_dedup if r["projection"] == proj]
        proj_all = proj_kmeans + proj_int

        # Pareto: non-dominated within projection
        pareto = [r for r in proj_all if not is_pareto_dominated(r, proj_all)]
        # Sort by bpw
        pareto.sort(key=lambda r: r["bits_per_weight"])
        pareto_rows.extend(pareto)

        print(f"\n--- {proj} Pareto frontier ({len(pareto)} configs) ---")
        print(f"{'scheme':<60} {'bpw':>7} {'err':>10} {'CR':>6}")
        for r in pareto:
            tag = "kmeans" if r["scheme"].startswith("kmeans_") else "int"
            print(
                f"{r['scheme']:<60} {r['bits_per_weight']:>7.3f} "
                f"{r['rel_fro_err']:>10.6f} {r['compression_ratio']:>6.2f} [{tag}]"
            )

    with open(args.pareto_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for r in pareto_rows:
            writer.writerow({k: r.get(k, "") for k in FIELDNAMES})
    print(f"\nWrote {args.pareto_out} ({len(pareto_rows)} rows)")

    # Best kmeans per bpw bucket
    print("\n=== Best kmeans per bpw bucket ===")
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
    for proj in projections:
        print(f"\n--- {proj} ---")
        proj_kmeans = [r for r in kmeans_rows if r["projection"] == proj]
        proj_int = [r for r in int_dedup if r["projection"] == proj]
        print(
            f"{'bucket':<12} {'best_kmeans':>10} {'@bpw':>7} "
            f"{'best_int':>10} {'@bpw':>7} {'winner':>8}"
        )
        for lo, hi in buckets:
            bk = [r for r in proj_kmeans if lo <= r["bits_per_weight"] < hi]
            bi = [r for r in proj_int if lo <= r["bits_per_weight"] < hi]
            if not bk:
                print(f"[{lo:.1f}-{hi:.1f})    {'-':>10} {'-':>7} ", end="")
            else:
                best_k = min(bk, key=lambda r: r["rel_fro_err"])
                print(
                    f"[{lo:.1f}-{hi:.1f})    {best_k['rel_fro_err']:>10.6f} "
                    f"{best_k['bits_per_weight']:>7.3f} ",
                    end="",
                )
            if not bi:
                print(f"{'-':>10} {'-':>7} ", end="")
            else:
                best_i = min(bi, key=lambda r: r["rel_fro_err"])
                print(f"{best_i['rel_fro_err']:>10.6f} {best_i['bits_per_weight']:>7.3f} ", end="")
            if bk and bi:
                bk_best = min(bk, key=lambda r: r["rel_fro_err"])
                bi_best = min(bi, key=lambda r: r["rel_fro_err"])
                winner = "kmeans" if bk_best["rel_fro_err"] < bi_best["rel_fro_err"] else "int"
                print(f"{winner:>8}")
            else:
                print()


if __name__ == "__main__":
    main()
