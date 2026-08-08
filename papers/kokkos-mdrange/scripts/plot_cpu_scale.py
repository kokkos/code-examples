#!/usr/bin/env python3
"""

Compare MDRangePolicy benchmark results from multiple Google Benchmark
JSON files. Plots bandwidth (GB/s) for increasing number of threads for
a given kernel and rank. Possible to plot more than one set of results
in the same plot.

Usage:
    python plot_cpu_scale.py info_file.txt

Number of json files should be twice (one set each for Stencil and Stream)
the number of threads in NUM_THREADS.
They should be specified as pairs (baseline, proposed) in proper order.

How an example info_file.txt looks like:
Stencil
2
omp_deftiles_t1_iter300_rep1_stencil_520.json
omp_deftiles_t1_iter300_rep1_stencil_auto_vect.json
omp_deftiles_t2_iter300_rep1_stencil_520.json
omp_deftiles_t2_iter300_rep1_stencil_auto_vect.json
omp_deftiles_t4_iter300_rep1_stencil_520.json
omp_deftiles_t4_iter300_rep1_stencil_auto_vect.json
omp_deftiles_t8_iter300_rep1_stencil_520.json
omp_deftiles_t8_iter300_rep1_stencil_auto_vect.json
omp_deftiles_t16_iter300_rep1_stencil_520.json
omp_deftiles_t16_iter300_rep1_stencil_auto_vect.json
omp_deftiles_t32_iter300_rep1_stencil_520.json
omp_deftiles_t32_iter300_rep1_stencil_auto_vect.json
Triad
5
omp_deftiles_t1_iter300_rep1_stream_520.json
omp_deftiles_t1_iter300_rep1_stream_auto_vect.json
omp_deftiles_t2_iter300_rep1_stream_520.json
omp_deftiles_t2_iter300_rep1_stream_auto_vect.json
omp_deftiles_t4_iter300_rep1_stream_520.json
omp_deftiles_t4_iter300_rep1_stream_auto_vect.json
omp_deftiles_t8_iter300_rep1_stream_520.json
omp_deftiles_t8_iter300_rep1_stream_auto_vect.json
omp_deftiles_t16_iter300_rep1_stream_520.json
omp_deftiles_t16_iter300_rep1_stream_auto_vect.json
omp_deftiles_t32_iter300_rep1_stream_520.json
omp_deftiles_t32_iter300_rep1_stream_auto_vect.json

"""

import json
import sys
import re
import os
import argparse
from collections import OrderedDict, defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scienceplots

# ── Configuration ────────────────────────────────────────────────────────────
NUM_THREADS = [1, 2, 4, 8, 16, 32]

STENCIL_RANKS_SIZES = {
    2: [10240],
    3: [512],
    4: [96]
}

REFERENCE_COLOR = "#9AA0A6"          # neutral grey for the Base reference bar
# Optional: override automatic labels (one per file, in order)
LABELS = ["Baseline", "With Auto-Vect"]
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


def extract_metrics(data, kernel, rank, num_threads):
    metrics = {kernel : {}}
    if kernel in ["Set", "Scale", "Copy", "Add", "Triad"]:
        for bench in data["benchmarks"]:
            name = bench["name"]
            pattern = rf"MDRangePolicy_{kernel}<(\d+)>"
            m = re.search(pattern, name)
            if m:
                cur_rank = int(m.group(1))
                if cur_rank == rank:
                    metrics[kernel][num_threads] = {
                        "gb_s": bench["FOM: GB/s"],
                        "time_ms": bench["real_time"],
                    }
    elif kernel == "Stencil":
        for bench in data["benchmarks"]:
            name = bench["name"]
            for target_size in STENCIL_RANKS_SIZES[rank]:
                pattern = rf"MDRangeStencil_{rank}D_MDRange_LayoutLeft/size:(\d+)/tile_size:(\S+)/manual_time"
                m = re.search(pattern, name)
                if m:
                    size = int(m.group(1))
                    if size == target_size:
                        metrics[kernel][num_threads] = {
                            "time_ms": bench["real_time"]
                        }
    else:
        sys.exit(f"Kernel '{kernel_arg}' not supported. Available kernels: {sorted(KERNELS)}")

    return metrics

def resolve_kernel(kernel_arg):
    KERNELS = ("Set", "Scale", "Copy", "Add", "Triad", "Stencil")
    k_lower = kernel_arg.lower()
    for k in KERNELS:
        if k_lower == k.lower():
            return k
    sys.exit(f"Kernel '{kernel_arg}' not found. Available kernels: {sorted(KERNELS)}")


def plot_scaling(all_metrics):
    plot_data = OrderedDict()
    for data in all_metrics:
        for bench, d in data.items():
            if bench not in plot_data:
                plot_data[bench] = {'rank': None, 'times': defaultdict(list)}
            if 'rank' in d:
                plot_data[bench]['rank'] = d['rank']
            else:
                for rank, metrics in d.items():
                    plot_data[bench]['times'][rank].append(metrics['time_ms'])

    fig, axes = plt.subplots(1, 2, figsize=(4, 2.75))

    print(plot_data)

    for ax, (bench, info) in zip(axes, plot_data.items()):
        ratios = [info['times'][t][0] / info['times'][t][1] for t in NUM_THREADS]

        print(ratios)

        ax.bar([str(t) for t in NUM_THREADS], ratios, color='green')
        ax.set_title(f"{bench}: Rank {info['rank']}")
        ax.set_xlabel("Threads")
        ax.set_ylabel("Speedup vs Baseline")
        ax.axhline(1.0, color='gray', linewidth=0.8, linestyle='--')

        for i, r in enumerate(ratios):
            ax.text(i, r, f"{r:.2f}", ha='center', va='bottom', fontsize=8)

    fig.tight_layout()

    return fig


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} info_file.txt")
        sys.exit(1)

    # Parsing the arguments with minimal error checking

    kernels = []
    ranks = []
    files = [[]]

    with open(sys.argv[1]) as info_file:
        file_lines = info_file.readlines()

        kernels.append(file_lines[0].strip())
        ranks.append(int(file_lines[1].strip()))

        argc = 2
        listc = 0

        while (argc < len(file_lines)):
            fp = file_lines[argc].strip()
            if fp.endswith(".json"):
                files[listc].append(fp)
            else:
                kernels.append(file_lines[argc].strip())
                argc = argc + 1
                ranks.append(int(file_lines[argc].strip()))
                listc = listc + 1
                files.append([])
            argc = argc + 1

    num_sets = len(files)
    num_settings = len(NUM_THREADS)

    input_error = False

    for n in range(num_sets):
        if (len(files[n]) != 2 * len(NUM_THREADS)):
            print(f"Enter {num_settings} pairs of JSON files for {kernels[n]}-{ranks[n]}")
            input_error = True

    if input_error:
        sys.exit(1)

    all_metrics = []
    klow = ""

    for n in range(num_sets):
        kernel = resolve_kernel(kernels[n])
        klow += kernel.lower() + "_"
        rank = ranks[n]

        all_metrics.append({kernel: {"rank": rank}})

        cur_files = files[n]

        for t in range(len(NUM_THREADS)):
            data = load_benchmarks(cur_files[2*t])
            metrics = extract_metrics(data, kernel, rank, NUM_THREADS[t])
            all_metrics.append(metrics)
            data = load_benchmarks(cur_files[2*t+1])
            metrics = extract_metrics(data, kernel, rank, NUM_THREADS[t])
            all_metrics.append(metrics)

    fig1 = plot_scaling(all_metrics)
    out1 = f"mdrange_{klow}scale.png"
    fig1.savefig(out1, dpi=600, bbox_inches="tight")
    print(f"Saved: {out1}")

if __name__ == "__main__":
    main()
