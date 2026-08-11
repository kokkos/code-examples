#!/usr/bin/env python3
"""

Compare MDRangeStencil benchmark results from multiple Google Benchmark JSON files.
Plots execution time (ms) for MDRange LayoutLeft and LayoutRight, grouped by problem size.

The arguments should be as follows:
1) All Json files in a particular order
2) Corresponding labels in the same order 

Usage:
    python plot_cpu_stencil.py file1.json file2.json label1 label2
"""

import json
import sys
import re
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
import numpy as np
import scienceplots

# ── Configuration ────────────────────────────────────────────────────────────
VARIANTS = ["MDRange_LayoutLeft", "MDRange_LayoutRight"]
RANKS_SIZES = {
    2: [10240],
    3: [512],
    4: [96]
}

# Colour encodes the file, hatch encodes the layout (same order as VARIANTS).
HATCHES = ["", "///"]  # LayoutLeft = solid, LayoutRight = hatched

plt.style.use(['science', 'ieee', 'std-colors'])

def load_benchmarks(filepath):
    with open(filepath) as f:
        return json.load(f)

def extract_metrics(data, variants, ranks_sizes):
    """
    Returns dict: {variant: {rank: {size: {"time_ms": ..., "tiles": [...]}}}}
    """
    metrics = {v: {r: {} for r in ranks_sizes} for v in variants}
    for bench in data["benchmarks"]:
        name = bench["name"]
        for variant in variants:
            for rank, sizes in ranks_sizes.items():
                pattern = rf"MDRangeStencil_{rank}D_{variant}/size:(\d+)/tile_size:(\S+)/manual_time"
                m = re.search(pattern, name)
                if m:
                    size = int(m.group(1))
                    if size in sizes:
                        tiles = []
                        for ti in range(rank):
                            key = f"tile_{ti}"
                            if key in bench:
                                tiles.append(int(bench[key]))
                        metrics[variant][rank][size] = {
                            "time_ms": bench["real_time"],
                            "tiles": tiles,
                        }
    return metrics


def _colors():
    return plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _legend_handles(file_labels, file_indices):
    """Colour patches for the files + hatch patches for the layouts."""
    colors = _colors()
    file_handles = [
        Patch(facecolor=colors[fi % len(colors)], edgecolor="black",
              linewidth=0.4, label=lab)
        for lab, fi in zip(file_labels, file_indices)
    ]
    layout_handles = [
        Patch(facecolor="white", edgecolor="black", linewidth=0.4,
              hatch=HATCHES[li % len(HATCHES)],
              label=VARIANTS[li].replace("MDRange_", ""))
        for li in range(len(VARIANTS))
    ]
    return file_handles + layout_handles


def plot_comparison(all_metrics, labels):
    """Single axes: file = colour, layout = hatch, grouped by rank."""
    n_files = len(all_metrics)
    n_layouts = len(VARIANTS)
    n_bars = n_files * n_layouts
    ranks = sorted(RANKS_SIZES.keys())

    fig, axe = plt.subplots(1, 1, figsize=(4, 2.75))
    colors = _colors()

    x = np.arange(len(ranks))
    bar_width = 0.8 / n_bars
    xlabels = [f"{r}D ({RANKS_SIZES[r][0]})" for r in ranks]

    for fi, metrics in enumerate(all_metrics):
        for li, variant in enumerate(VARIANTS):
            values = [
                metrics[variant][r].get(RANKS_SIZES[r][0], {}).get("time_ms", 0)
                for r in ranks
            ]
            bi = fi * n_layouts + li
            offset = (bi - (n_bars - 1) / 2) * bar_width
            axe.bar(
                x + offset, values,
                width=bar_width * 0.9,
                color=colors[fi % len(colors)],
                hatch=HATCHES[li % len(HATCHES)],
                edgecolor="black",
                linewidth=0.4,
                zorder=1,
            )

    axe.set_xlabel("Rank (problem size per dim)")
    axe.set_xticks(x)
    axe.set_xticklabels(xlabels)
    axe.tick_params(axis="x", which="both", bottom=False)
    axe.set_ylabel("Time (ms)")
    axe.grid(axis="y", alpha=0.4, zorder=0)
    axe.set_axisbelow(True)

    handles = _legend_handles(labels[:n_files], range(n_files))
    fig.tight_layout(rect=[0, 0, 1, 0.85])
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=True,
        framealpha=0.8,
        columnspacing=1.0,
        handlelength=1.2,
        fontsize=7,
    )
    return fig


def plot_speedup(all_metrics, labels):
    """Single axes: file = colour, layout = hatch, grouped by rank."""
    if len(all_metrics) < 2:
        return None

    n_series = len(all_metrics) - 1          # non-baseline files
    n_layouts = len(VARIANTS)
    n_bars = n_series * n_layouts
    ranks = sorted(RANKS_SIZES.keys())

    fig, axe = plt.subplots(1, 1, figsize=(4, 2.75))
    colors = _colors()

    x = np.arange(len(ranks))
    bar_width = 0.85 / n_bars
    xlabels = [f"{r}D ({RANKS_SIZES[r][0]})" for r in ranks]

    max_bar_height = 0
    for si in range(n_series):
        fi = si + 1                          # actual file index
        for li, variant in enumerate(VARIANTS):
            baseline = all_metrics[0][variant]
            speedups = []
            for r in ranks:
                s = RANKS_SIZES[r][0]
                base_t = baseline[r].get(s, {}).get("time_ms", 1)
                new_t = all_metrics[fi][variant][r].get(s, {}).get("time_ms", 0)
                speedups.append(base_t / new_t if new_t else 0)

            bi = si * n_layouts + li
            offset = (bi - (n_bars - 1) / 2) * bar_width
            bars = axe.bar(
                x + offset, speedups,
                width=bar_width * 0.75,
                color=colors[(fi - 1) % len(colors)],
                hatch=HATCHES[li % len(HATCHES)],
                edgecolor="black",
                linewidth=0.3,
                zorder=2,
            )

            for bar, sp in zip(bars, speedups):
                max_bar_height = max(max_bar_height, bar.get_height())
                axe.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{sp:.2f}×",
                    ha="center", va="bottom", fontsize=6, rotation=0,
                )

    axe.axhline(y=1.0, color="grey", linestyle="--", linewidth=1.0, alpha=1.0, zorder=1)
    axe.set_xlabel("Rank (problem size per dim)")
    axe.set_xticks(x)
    axe.set_xticklabels(xlabels)
    axe.tick_params(axis="x", which="both", bottom=False)
    axe.set_ylabel(f"Speedup vs {labels[0]}")
    axe.grid(axis="y", alpha=0.4, zorder=0)
    axe.set_axisbelow(True)
    axe.set_ylim(top=max_bar_height * 1.15)
    handles = _legend_handles(labels[1:n_series + 1], range(0, n_series))
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=True,
        framealpha=0.8,
        columnspacing=1.0,
        handlelength=1.2,
        fontsize=7,
    )
    return fig

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} file1.json [file2.json] label1 [label2]")
        sys.exit(1)

    files_labels = sys.argv[1:]
    if (len(files_labels) % 2 != 0):
        print("Enter the same number of json files and labels")
        print(f"Usage: {sys.argv[0]} file1.json [file2.json] label1 [label2]")
        sys.exit(1)

    num_cases = int(len(files_labels)/2)

    filepaths = files_labels[:num_cases]

    labels = files_labels[num_cases:]

    for fp in filepaths:
        if not fp.endswith(".json"):
            print("Enter the same number of json files and labels")
            print(f"Usage: {sys.argv[0]} file1.json [file2.json] label1 [label2]")
            sys.exit(1)

    for label in labels:
        if label.endswith(".json"):
            print("Enter the same number of json files and labels")
            print(f"Usage: {sys.argv[0]} file1.json [file2.json] label1 [label2]")
            sys.exit(1)

    all_metrics = []
    for fp in filepaths:
        data = load_benchmarks(fp)
        metrics = extract_metrics(data, VARIANTS, RANKS_SIZES)
        all_metrics.append(metrics)

    fig1 = plot_comparison(all_metrics, labels)
    fig1.savefig("mdrange_stencil_time.png", dpi=600, bbox_inches="tight")
    print("Saved: mdrange_stencil_time.png")

    fig2 = plot_speedup(all_metrics, labels)
    if fig2 is not None:
        fig2.savefig("mdrange_stencil_speedup.png", dpi=600, bbox_inches="tight")
        print("Saved: mdrange_stencil_speedup.png")

    # Avoid showing plots in interactive mode
    # plt.show()


if __name__ == "__main__":
    main()
