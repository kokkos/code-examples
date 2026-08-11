#!/usr/bin/env python3
"""

Compare a single Kokkos variant (default: Kokkos_Lambda) across several run
configurations, using RAJAPerf / polybench CSV output.

Usage:
    python plot_gpu_rajaPerf.py \
        .../heat_3d_80000000-kernel-run-data.csv \
        .../hydro_2d_40000000-kernel-run-data.csv \
        .../heat_3d_80000000-kernel-run-data.csv \
        .../hydro_2d_40000000-kernel-run-data.csv \
        .../heat_3d_80000000-kernel-run-data.csv \
        .../hydro_2d_40000000-kernel-run-data.csv
"""

import csv
import re
import sys
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

plt.style.use(['science', 'ieee', 'std-colors'])

# ── Configuration ────────────────────────────────────────────────────────────
KOKKOS_VARIANT = "Kokkos_Lambda"     # the variant compared across configs
REFERENCE_VARIANT = None             # force a reference, e.g. "Base_CUDA";
                                     # None = auto-detect from the list below
REFERENCE_CANDIDATES = ["Base_CUDA", "Base_HIP", "Base_SYCL", "Base_OpenMP"]

METRICS = ["bandwidth", "flops"]     # which metrics to plot (subplots)
KERNELS = []                         # [] = auto-detect (order the files are passed)

# Configurations to compare, HARD-CODED and IN ORDER. The first one is the
# baseline for the speedup. Files are passed config by config (this many files
# per config = len(filepaths) // len(CONFIGS)).
CONFIGS = ["5.0.2", "5.1.0", "5.2.0"]

STRIP_KERNEL_PREFIX = True           # drop Polybench_/Apps_/Basic_/Lcals_ in x labels

REFERENCE_COLOR = "#9AA0A6"          # neutral grey for the Base reference bar

METRIC_INFO = {
    "time": {"needles": ("mean time",), "ylabel": "Time / rep (s)",
             "title": "Mean time per rep", "lower_better": True},
    "bandwidth": {"needles": ("bandwidth",), "ylabel": "Bandwidth (GiB/s)",
                  "title": "Bandwidth", "lower_better": False},
    "flops": {"needles": ("flop",), "ylabel": "Performance (GFLOP/s)",
              "title": "FLOP/s", "lower_better": False},
}


# ── Parsing ──────────────────────────────────────────────────────────────────

def find_col(header, *needles):
    for h in header:
        if all(n.lower() in h.lower() for n in needles):
            return h
    return None


def to_float(s):
    try:
        return float(s.strip())
    except (ValueError, AttributeError):
        return float("nan")


def load_csv(filepath):
    """Read a RAJAPerf-style CSV, skipping any preamble before the header row."""
    with open(filepath, newline="") as f:
        lines = f.readlines()

    header_idx = next(
        (i for i, ln in enumerate(lines)
         if "Kernel" in ln and "Variant" in ln and "Mean time per rep" in ln),
        None,
    )
    if header_idx is None:
        raise ValueError(f"Could not locate a header row in {filepath}")

    reader = csv.reader(lines[header_idx:])
    header = [h.strip() for h in next(reader)]
    cols = {
        "kernel": find_col(header, "kernel"),
        "variant": find_col(header, "variant"),
        "time": find_col(header, "mean time"),
        "bandwidth": find_col(header, "bandwidth"),
        "flops": find_col(header, "flop"),
    }
    rows = [[c.strip() for c in raw] for raw in reader
            if raw and any(c.strip() for c in raw) and len(raw) >= len(header)]
    return header, cols, rows


def extract_metrics(header, cols, rows):
    """Returns {kernel: {variant: {"time":.., "bandwidth":.., "flops":..}}}."""
    idx = {k: header.index(v) for k, v in cols.items() if v is not None}
    metrics = {}
    for row in rows:
        variant = row[idx["variant"]]
        kernel = row[idx["kernel"]]
        per_kernel = metrics.setdefault(kernel, {})
        if variant in per_kernel:
            continue  # keep first match
        per_kernel[variant] = {
            m: to_float(row[idx[m]]) if m in idx else float("nan")
            for m in ("time", "bandwidth", "flops")
        }
    return metrics


def prettify_base(name):
    return re.sub(r"^(Base)_", "Native ", name)

def prettify_kernel(name):
    if STRIP_KERNEL_PREFIX:
        return re.sub(r"^(Polybench|Apps|Basic|Lcals|Stream|Algorithm)_", "", name)
    return name

def ordered_unique(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ── Lookups ──────────────────────────────────────────────────────────────────

def get(data, config, kernel, variant, metric):
    return data.get(config, {}).get(kernel, {}).get(variant, {}).get(metric, float("nan"))


def reference_value(data, configs, kernel, ref_variant, metric):
    """Base reference is config-independent: take the first config that has it."""
    for cfg in configs:
        v = get(data, cfg, kernel, ref_variant, metric)
        if np.isfinite(v):
            return v
    return float("nan")


def detect_reference(data, configs):
    if REFERENCE_VARIANT:
        return REFERENCE_VARIANT
    present = {v for cfg in configs for k in data[cfg] for v in data[cfg][k]}
    return next((c for c in REFERENCE_CANDIDATES if c in present), None)


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_comparison(data, configs, kernels, ref_variant, metric):
    info = METRIC_INFO[metric]
    n_bars = len(configs) + (1 if ref_variant else 0)
    xticklabels = [prettify_kernel(k) for k in kernels]
    x = np.arange(len(kernels))

    fig, ax = plt.subplots(figsize=(4, 2.75))

    slot = 0
    width = 0.8 / max(n_bars, 1)
    if ref_variant:
        ref_vals = [reference_value(data, configs, k, ref_variant, metric)
                    for k in kernels]
        offset = (slot - (n_bars - 1) / 2) * width
        ax.bar(x + offset, ref_vals, width=width * 0.9,
               label=f"{prettify_base(ref_variant)} (ref)", color=REFERENCE_COLOR,
               edgecolor="white", linewidth=0.6, hatch="///", zorder=1)
        slot += 1

    for i, cfg in enumerate(configs):
        vals = [get(data, cfg, k, KOKKOS_VARIANT, metric) for k in kernels]
        offset = (slot - (n_bars - 1) / 2) * width
        ax.bar(x + offset, vals, width=width * 0.9, label=cfg, linewidth=0.6, zorder=1)
        slot += 1

    # ax.set_title(f"Kokkos across configs - {info['title']}"
    #              + (f"   (ref: {prettify_base(ref_variant)})" if ref_variant else ""),
    #              fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel(info["ylabel"])
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)
    ax.tick_params(axis="x", which="both", bottom=False)
    ax.grid(axis="y", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        ncol=n_bars,
        frameon=True,
        framealpha=0.8,
        columnspacing=1.0,
        handlelength=1.2,
    )

    fig.tight_layout()
    return fig


def plot_speedup(data, configs, kernels, metric):
    if len(configs) < 2:
        return None

    info = METRIC_INFO[metric]
    lower_better = info["lower_better"]
    base_cfg = configs[0]
    others = configs[1:]
    xticklabels = [prettify_kernel(k) for k in kernels]
    x = np.arange(len(kernels))
    width = 0.8 / len(others)

    max_bar_height = 0
    fig, ax = plt.subplots(1,1, figsize=(4, 2.75))

    for i, cfg in enumerate(others):
        speedups = []
        for k in kernels:
            base = get(data, base_cfg, k, KOKKOS_VARIANT, metric)
            new = get(data, cfg, k, KOKKOS_VARIANT, metric)
            if not (np.isfinite(base) and np.isfinite(new)) or base == 0 or new == 0:
                speedups.append(float("nan"))
            else:
                speedups.append(base / new if lower_better else new / base)

        offset = (i - (len(others) - 1) / 2) * width
        bars = ax.bar(x + offset, speedups, width=width * 0.9,
                      label=f"{cfg} / {base_cfg}",
                      edgecolor="white", linewidth=0.6, zorder=1)
        for bar, sp in zip(bars, speedups):
            if bar.get_height() > max_bar_height:
                max_bar_height = bar.get_height()
            if np.isfinite(sp):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.05, f"{sp:.2f}×",
                        ha="center", va="bottom", fontsize=7)

    ax.axhline(y=1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)
    # ax.set_title(f"Kokkos speedup vs {base_cfg} - {info['title']}"
    #              "   (higher = better)",
    #              fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel(f"Speedup vs {base_cfg}")
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(top=max_bar_height * 1.15)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        ncol=len(others),
        frameon=True,
        framealpha=0.8,
        columnspacing=1.0,
        handlelength=1.2,
    )

    fig.tight_layout()
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <files, config by config, kernels in order>")
        print(f"Configs (edit CONFIGS in the script): {CONFIGS}")
        sys.exit(1)

    filepaths = sys.argv[1:]
    bad = [m for m in METRICS if m not in METRIC_INFO]
    if bad:
        print(f"Unknown metric(s) {bad}. Choose from {list(METRIC_INFO)}.")
        sys.exit(1)

    n_cfg = len(CONFIGS)
    if len(filepaths) % n_cfg != 0:
        print(f"Error: {len(filepaths)} files for {n_cfg} configs {CONFIGS}; "
              f"the count must be a multiple of {n_cfg} - pass the same kernels, "
              f"in the same order, for every config.")
        sys.exit(1)
    per_config = len(filepaths) // n_cfg

    data, configs = {}, list(CONFIGS)
    for ci, cfg in enumerate(CONFIGS):
        data[cfg] = {}
        for fp in filepaths[ci * per_config:(ci + 1) * per_config]:
            header, cols, rows = load_csv(fp)
            for kernel, variants in extract_metrics(header, cols, rows).items():
                dk = data[cfg].setdefault(kernel, {})
                for v, vals in variants.items():
                    dk.setdefault(v, vals)  # keep first occurrence

    kernels = KERNELS or ordered_unique(k for cfg in configs for k in data[cfg])
    ref_variant = detect_reference(data, configs)

    print(f"Configs : {configs}")
    print(f"Kernels : {kernels}")
    print(f"Compared: {KOKKOS_VARIANT}   |   reference: {ref_variant}")
    if ref_variant is None:
        print("  (no Base_* reference found - plotting Kokkos bars only)")

    for metric in METRICS:
        fig = plot_comparison(data, configs, kernels, ref_variant, metric)
        out = f"rajaperf_compare_{metric}.png"
        fig.savefig(out, dpi=600, bbox_inches="tight")
        print(f"Saved: {out}")

    for metric in METRICS:
        fig = plot_speedup(data, configs, kernels, metric)
        if fig:
            out = f"rajaperf_speedup_{metric}.png"
            fig.savefig(out, dpi=600, bbox_inches="tight")
            print(f"Saved: {out}")

    # Avoid showing plots in interactive mode
    # plt.show()



if __name__ == "__main__":
    main()
