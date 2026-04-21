"""Compute dataset statistics from AutoMate parquet files.

Produces reports/stats.json and reports/stats.html with plots.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def plot_hist(values, title, xlabel, log_x=False, bins=60):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    fig, ax = plt.subplots(figsize=(7, 4))
    if log_x:
        values = values[values > 0]
        ax.hist(values, bins=np.logspace(np.log10(values.min()), np.log10(values.max()), bins))
        ax.set_xscale("log")
    else:
        ax.hist(values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(alpha=0.3)
    return fig


def plot_bar(labels, counts, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    order = np.argsort(counts)[::-1]
    labels = [labels[i] for i in order]
    counts = [counts[i] for i in order]
    ax.bar(range(len(labels)), counts)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel("count")
    ax.grid(alpha=0.3, axis="y")
    return fig


def plot_scatter(x, y, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=4, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data/automate"))
    p.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = p.parse_args()

    data_dir: Path = args.data_dir.resolve()
    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    assemblies = pd.read_parquet(data_dir / "assemblies.parquet")
    parts = pd.read_parquet(data_dir / "parts.parquet")
    mates = pd.read_parquet(data_dir / "mates.parquet")

    print(f"assemblies: {len(assemblies):,} rows, columns={list(assemblies.columns)[:8]}...")
    print(f"parts:      {len(parts):,} rows")
    print(f"mates:      {len(mates):,} rows")

    summary: dict = {
        "n_assemblies": len(assemblies),
        "n_parts_unique": len(parts),
        "n_mates": len(mates),
    }

    # Key distributions from assemblies.parquet
    if "n_parts" in assemblies.columns:
        summary["parts_per_assembly"] = {
            "min": int(assemblies["n_parts"].min()),
            "max": int(assemblies["n_parts"].max()),
            "mean": float(assemblies["n_parts"].mean()),
            "median": float(assemblies["n_parts"].median()),
            "p95": float(assemblies["n_parts"].quantile(0.95)),
        }
    if "n_mates" in assemblies.columns:
        summary["mates_per_assembly"] = {
            "min": int(assemblies["n_mates"].min()),
            "max": int(assemblies["n_mates"].max()),
            "mean": float(assemblies["n_mates"].mean()),
            "median": float(assemblies["n_mates"].median()),
        }
    if {"n_step", "n_parts"}.issubset(assemblies.columns):
        summary["pct_full_step"] = float((assemblies["n_step"] == assemblies["n_parts"]).mean())
    elif "has_all_step" in assemblies.columns:
        summary["pct_full_step"] = float(assemblies["has_all_step"].mean())

    # Mate type histogram
    mate_type_counts = mates["mateType"].value_counts()
    summary["mate_types"] = {k: int(v) for k, v in mate_type_counts.items()}

    # Part reuse
    if "parts" in mates.columns:
        # mates.parts is list of 2 part_ids per row; count across all assemblies
        pass
    if "n_parts" in assemblies.columns:
        summary["top10_largest"] = [
            {"id": row.get("assemblyId") or row.get(assemblies.columns[0]),
             "n_parts": int(row["n_parts"]),
             "n_mates": int(row["n_mates"]) if "n_mates" in row else None}
            for _, row in assemblies.nlargest(10, "n_parts").iterrows()
        ]

    # Bounding box stats from parts.parquet
    bb_diag = None
    if {"bb_0", "bb_1", "bb_2", "bb_3", "bb_4", "bb_5"}.issubset(parts.columns):
        bb_diag = np.sqrt(
            (parts["bb_3"] - parts["bb_0"]) ** 2
            + (parts["bb_4"] - parts["bb_1"]) ** 2
            + (parts["bb_5"] - parts["bb_2"]) ** 2
        )
        summary["part_bbox_diag_m"] = {
            "min": float(bb_diag.min()),
            "max": float(bb_diag.max()),
            "median": float(bb_diag.median()),
            "p95": float(bb_diag.quantile(0.95)),
        }

    # ---------------- plots ----------------
    plots: dict[str, str] = {}
    if "n_parts" in assemblies.columns:
        plots["parts_per_assembly"] = fig_to_b64(
            plot_hist(assemblies["n_parts"], "Parts per assembly (log x)", "n_parts", log_x=True)
        )
    if "n_mates" in assemblies.columns:
        plots["mates_per_assembly"] = fig_to_b64(
            plot_hist(assemblies["n_mates"], "Mates per assembly (log x)", "n_mates", log_x=True)
        )
    plots["mate_types"] = fig_to_b64(
        plot_bar(list(mate_type_counts.index), list(mate_type_counts.values), "Mate type distribution")
    )
    if {"n_parts", "n_mates"}.issubset(assemblies.columns):
        plots["complexity_scatter"] = fig_to_b64(
            plot_scatter(
                assemblies["n_parts"].clip(lower=1),
                assemblies["n_mates"].clip(lower=1),
                "Assembly complexity: parts vs mates",
                "n_parts (log)",
                "n_mates (log)",
            )
        )
    if bb_diag is not None:
        plots["part_bbox_diag"] = fig_to_b64(
            plot_hist(bb_diag.replace(0, np.nan), "Part bbox diagonal (m, log x)", "meters", log_x=True)
        )

    # ---------------- write outputs ----------------
    stats_json_path = out_dir / "stats.json"
    stats_json_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Wrote {stats_json_path}")

    html = ["<!doctype html><meta charset='utf-8'><title>AutoMate stats</title>",
            "<style>body{font-family:ui-sans-serif,system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem}",
            "h1,h2{border-bottom:1px solid #ddd;padding-bottom:0.3rem}",
            "pre{background:#f4f4f4;padding:0.8rem;border-radius:4px;overflow-x:auto}",
            "img{max-width:100%;border:1px solid #ddd;margin:0.5rem 0}</style>",
            "<h1>AutoMate dataset statistics</h1>",
            f"<p>Source: <code>{data_dir}</code></p>",
            "<h2>Summary</h2>",
            f"<pre>{json.dumps(summary, indent=2, default=str)}</pre>"]
    for name, b64 in plots.items():
        html.append(f"<h2>{name}</h2><img src='data:image/png;base64,{b64}'/>")
    stats_html_path = out_dir / "stats.html"
    stats_html_path.write_text("\n".join(html))
    print(f"Wrote {stats_html_path}")


if __name__ == "__main__":
    main()
