#!/usr/bin/env python3
"""

Compare MDRangePolicy benchmark results from multiple Google Benchmark JSON files.
Plots bandwidth (GB/s) for Copy and Add kernels, grouped by rank (1-6).

Usage:
    python plot_gpu_stream.py file1.json file2.json [file3.json]

Each file gets a label derived from its filename (or you can customize LABELS below).
"""

import json
import sys
import re
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scienceplots

# ── Configuration ────────────────────────────────────────────────────────────
RANKS = [2, 3, 4, 5, 6]
REFERENCE_COLOR = "#9AA0A6"          # neutral grey for the Base reference bar
# Optional: override automatic labels (one per file, in order)
LABELS = ["Baseline", "Refactored", "New tile size", "No grid-stride"]
#LABELS = None

plt.style.use(['science', 'ieee', 'std-colors'])

def load_benchmarks(filepath):
    with open(filepath) as f:
        data = json.load(f)
    return data


def extract_label(filepath):
    base = os.path.splitext(os.path.basename(filepath))[0]
    # Trim common prefixes
    for prefix in ["benchmark_", "bench_", "result_", "test_"]:
        if base.lower().startswith(prefix):
            base = base[len(prefix):]
    return base


def extract_metrics(data, kernel, ranks):
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


def plot_comparison(all_metrics, labels, kernel, metric_key="gb_s", ylabel="Bandwidth (GB/s)"):
    n_files = len(all_metrics)

    fig, axe = plt.subplots(1, 1, figsize=(4, 2.75))

    bar_width = 0.8 / n_files
    x = np.arange(len(RANKS))

    for i, (metrics, label) in enumerate(zip(all_metrics, labels)):
        values = [metrics[kernel].get(r, {}).get(metric_key, 0) for r in RANKS]
        offset = (i - (n_files - 1) / 2) * bar_width
        bars = axe.bar(
            x + offset, values,
            width=bar_width * 0.85,
            label=label,
            linewidth=0.6,
            color=REFERENCE_COLOR if i == 0 else None,
            hatch="///" if i == 0 else None,
            edgecolor="white" if i == 0 else None,
            zorder=1,
        )

    # axe.set_title(f"MDRangePolicy - {kernel}", fontsize=13, fontweight="bold", pad=10)
    axe.set_xlabel("Rank (number of dimensions)")
    axe.set_xticks(x)
    axe.set_xticklabels([str(r) for r in RANKS])
    axe.set_ylabel(ylabel)
    axe.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axe.tick_params(axis="x", which="both", bottom=False)
    axe.grid(axis="y", alpha=0.4, zorder=0)
    axe.set_axisbelow(True)
    axe.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        ncol=n_files,
        frameon=True,
        framealpha=0.8,
        columnspacing=1.0,
        handlelength=1.2,
    )

    fig.tight_layout()
    return fig


def plot_speedup(all_metrics, labels, kernel):
    if len(all_metrics) < 2:
        return None

    n_series = len(all_metrics) - 1
    max_bar_height = 0

    fig, axe = plt.subplots(1, 1, figsize=(4, 2.75))

    bar_width = 0.9 / n_series
    x = np.arange(len(RANKS))

    baseline = all_metrics[0][kernel]
    for i in range(1, len(all_metrics)):
        speedups = []
        for r in RANKS:
            base_val = baseline.get(r, {}).get("gb_s", 1)
            new_val = all_metrics[i][kernel].get(r, {}).get("gb_s", 0)
            speedups.append(new_val / base_val if base_val else 0)

        offset = (i - 1 - (n_series - 1) / 2) * bar_width
        bars = axe.bar(
            x + offset, speedups,
            width=bar_width * 0.85,
            label=f"{labels[i]}",
            linewidth=0.6,
            zorder=2,
        )
        # Annotate values
        for bar, sp in zip(bars, speedups):
            if bar.get_height() > max_bar_height:
                max_bar_height = bar.get_height()
            axe.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f"{sp:.2f}×", ha="center", va="bottom", fontsize=6)

    axe.axhline(y=1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)
    # axe.set_title(f"Speedup - {kernel}", fontsize=13, fontweight="bold", pad=10)
    axe.set_xlabel("Rank (number of dimensions)")
    axe.set_xticks(x)
    axe.set_xticklabels([str(r) for r in RANKS])
    axe.tick_params(axis="x", which="both", bottom=False)
    axe.set_ylabel(f"Speedup vs {labels[0]}")
    axe.grid(axis="y", alpha=0.4, zorder=0)
    axe.set_axisbelow(True)
    axe.set_ylim(top=max_bar_height * 1.15)
    axe.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        ncol=n_series,
        frameon=True,
        framealpha=0.8,
        columnspacing=1.0,
        handlelength=1.2,
    )

    fig.tight_layout()
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot MDRangePolicy bandwidth/speedup for a single kernel."
    )
    parser.add_argument("kernel",  help="Kernel to plot (Set, Scale, Copy, Add, Triad)")
    parser.add_argument("files", nargs="+", help="Benchmark JSON files (up to 4)")
    args = parser.parse_args()

    filepaths = args.files
    kernel = resolve_kernel(args.kernel)
    if len(filepaths) > 4:
        print("Warning: only the first 4 files will be used.")
        filepaths = filepaths[:4]

    labels = LABELS if LABELS and len(LABELS) >= len(filepaths) else [
        extract_label(fp) for fp in filepaths
    ]

    all_metrics = []
    for fp in filepaths:
        data = load_benchmarks(fp)
        metrics = extract_metrics(data, kernel, RANKS)
        all_metrics.append(metrics)

    klow = kernel.lower()

    fig1 = plot_comparison(all_metrics, labels, kernel)
    out1 = f"mdrange_{klow}_bandwidth.png"
    fig1.savefig(out1, dpi=600, bbox_inches="tight")
    print(f"Saved: {out1}")

    fig2 = plot_speedup(all_metrics, labels, kernel)
    out2 = f"mdrange_{klow}_speedup.png"
    fig2.savefig(out2, dpi=600, bbox_inches="tight")
    print(f"Saved: {out2}")

    # Avoid showing plots in interactive mode
    # plt.show()


if __name__ == "__main__":
    main()
