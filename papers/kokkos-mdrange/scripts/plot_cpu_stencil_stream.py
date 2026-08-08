#!/usr/bin/env python3
"""

Plot MDRange benchmark results from Google Benchmark JSON files.
Stencil execution time and Stream bandwidth are displayed in separate panels
because they represent different physical quantities.

The arguments should be as follows:
1) All Json files in a particular order
2) Corresponding labels in the same order

Usage:
    python plot_cpu_stencil_stream.py stencil1.json stencil2.json stream1.json stream2.json label1 label2 stream_kernel

"""

import json
import re
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import scienceplots

# ── Configuration ────────────────────────────────────────────────────────────
# Stencil benchmarks
STENCIL_VARIANT = "MDRange_LayoutRight"
STENCIL_RANKS_SIZES = {
    2: [10240],
    3: [512],
    4: [96]
}

# Stream benchmarks
STREAM_RANKS = [5, 6]

# Same native width as the other single-column CPU figure, with extra height
# for readable side-by-side panels.
PAPER_FIGSIZE = (4.0, 3.2)
PAPER_DPI = 600
BAR_GROUP_WIDTH = 0.4
BAR_WIDTH_RATIO = 0.85
BAR_GROUP_SPACING = 0.65
BASELINE_COLOR = "#9AA0A6"
BASELINE_HATCH = "///"

paper_styles = ["science", "ieee", "std-colors"]
if not all(shutil.which(command) for command in ("latex", "dvipng")):
    paper_styles.append("no-latex")
plt.style.use(paper_styles)

if "no-latex" in paper_styles:
    # Nimbus Roman is the Times-compatible font used by the reference plots.
    plt.rcParams["font.serif"] = ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"]

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9.6,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

def load_benchmarks(filepath):
    with open(filepath) as f:
        return json.load(f)

def extract_stencil_metrics(data, variant, ranks_sizes):
    """
    Returns dict: {variant: {rank: {size: {"time_ms": ..., "tiles": [...]}}}}
    """
    metrics = {variant: {r: {} for r in ranks_sizes}}
    for bench in data["benchmarks"]:
        name = bench["name"]
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


def extract_stream_metrics(data, kernel, ranks):
    """
    Returns dict: {rank: {"gb_s": ..., "time_ms": ...}}}
    """
    metrics = {kernel : {}}
    for bench in data["benchmarks"]:
        name = bench["name"]
        pattern = rf"MDRangePolicy_{kernel}<(\d+)>"
        m = re.search(pattern, name)
        if m:
            rank = int(m.group(1))
            if rank in ranks:
                metrics[kernel][rank] = {
                    "gb_s": bench["FOM: GB/s"],
                    "time_ms": bench["real_time"],
                }
    return metrics


def resolve_kernel(kernel_arg):
    KERNELS = ("Set", "Scale", "Copy", "Add", "Triad")
    k_lower = kernel_arg.lower()
    for k in KERNELS:
        if k_lower == k.lower():
            return k
    sys.exit(f"Kernel '{kernel_arg}' not found. Available kernels: {sorted(KERNELS)}")


def _colors():
    return plt.rcParams["axes.prop_cycle"].by_key()["color"]


def plot_stencil(axe, stencil_metrics, labels):
    """Plot Stencil execution time using the paper's style."""
    n_series = len(labels)
    colors = _colors()

    ranks = sorted(STENCIL_RANKS_SIZES)
    x = np.arange(len(ranks)) * BAR_GROUP_SPACING
    bar_width = BAR_GROUP_WIDTH / n_series

    for fi, (metrics, label) in enumerate(zip(stencil_metrics, labels)):
        is_baseline = fi == 0
        values = [
            metrics[STENCIL_VARIANT][r].get(STENCIL_RANKS_SIZES[r][0], {}).get("time_ms", 0)
            for r in ranks
        ]
        offset = (fi - (n_series - 1) / 2) * bar_width
        axe.bar(
            x + offset, values,
            width=bar_width * BAR_WIDTH_RATIO,
            color=BASELINE_COLOR if is_baseline else colors[fi % len(colors)],
            hatch=BASELINE_HATCH if is_baseline else None,
            edgecolor="white" if is_baseline else "black",
            linewidth=0.4,
            zorder=2,
            label=label,
        )

    axe.set_xlabel("Rank (size per dim)")
    axe.set_xticks(x)
    axe.set_xticklabels([f"{r}D ({STENCIL_RANKS_SIZES[r][0]})" for r in ranks])
    axe.tick_params(axis="x", which="both", bottom=False)
    axe.set_ylabel("Time (ms)")
    axe.set_title("Stencil")
    axe.grid(axis="y", alpha=0.4, zorder=0)
    axe.set_axisbelow(True)


def plot_stream(
    axe,
    stream_metrics,
    labels,
    stream_kernel,
    metric_key="gb_s",
    ylabel="Bandwidth (GB/s)",
):
    """Plot Stream bandwidth using the paper's style."""
    n_series = len(labels)
    colors = _colors()

    x = np.arange(len(STREAM_RANKS)) * BAR_GROUP_SPACING
    bar_width = BAR_GROUP_WIDTH / n_series

    for i, (metrics, label) in enumerate(zip(stream_metrics, labels)):
        is_baseline = i == 0
        values = [
            metrics[stream_kernel].get(r, {}).get(metric_key, 0)
            for r in STREAM_RANKS
        ]
        offset = (i - (n_series - 1) / 2) * bar_width
        axe.bar(
            x + offset, values,
            width=bar_width * BAR_WIDTH_RATIO,
            color=BASELINE_COLOR if is_baseline else colors[i % len(colors)],
            hatch=BASELINE_HATCH if is_baseline else None,
            edgecolor="white" if is_baseline else "black",
            linewidth=0.4,
            zorder=2,
            label=label,
        )

    axe.set_xlabel("Rank")
    axe.set_xticks(x)
    axe.set_xticklabels([str(rank) for rank in STREAM_RANKS])
    axe.tick_params(axis="x", which="both", bottom=False)
    axe.set_ylabel(ylabel)
    axe.set_title(f"STREAM {stream_kernel}")
    axe.grid(axis="y", alpha=0.4, zorder=0)
    axe.set_axisbelow(True)


def plot_comparison(stencil_metrics, stream_metrics, labels, stream_kernel):
    """Plot Stencil and Stream in equal-sized panels with a shared legend."""
    fig, (stencil_axe, stream_axe) = plt.subplots(
        1,
        2,
        figsize=PAPER_FIGSIZE,
        gridspec_kw={"width_ratios": [1, 1]},
    )

    plot_stencil(stencil_axe, stencil_metrics, labels)
    plot_stream(stream_axe, stream_metrics, labels, stream_kernel)

    fig.tight_layout(rect=[0, 0, 1, 0.91], w_pad=1.2)
    handles, legend_labels = stencil_axe.get_legend_handles_labels()
    fig.legend(
        handles=handles,
        labels=legend_labels,
        loc="upper center",
        frameon=True,
        framealpha=0.8,
        columnspacing=1.0,
        handlelength=1.2,
        fontsize=7,
        ncol=len(labels),
    )
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 8:
        print(f"Usage: {sys.argv[0]} file1.json file2.json file3.json file4.json label1 label2 stream_kernel")
        sys.exit(1)

    files_labels = sys.argv[1:]

    stencil_paths = files_labels[0:2]

    stream_paths = files_labels[2:4]

    labels = files_labels[4:6]

    stream_kernel = resolve_kernel(files_labels[6])

    stencil_metrics = []
    for fp in stencil_paths:
        data = load_benchmarks(fp)
        metrics = extract_stencil_metrics(data, STENCIL_VARIANT, STENCIL_RANKS_SIZES)
        stencil_metrics.append(metrics)

    stream_metrics = []
    for fp in stream_paths:
        data = load_benchmarks(fp)
        metrics = extract_stream_metrics(data, stream_kernel, STREAM_RANKS)
        stream_metrics.append(metrics)

    figure = plot_comparison(
        stencil_metrics,
        stream_metrics,
        labels,
        stream_kernel,
    )
    figure.savefig(
        "mdrange_cpu_t8.png",
        dpi=PAPER_DPI,
        bbox_inches="tight",
    )
    print("Saved: mdrange_cpu_t8.png")


if __name__ == "__main__":
    main()
