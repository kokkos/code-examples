#!/bin/bash
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/.versions.sh"
CPU_GPU=""
BUILD_DIR=""
PLATFORM=""
NUM_THREADS=1

usage() {
  echo "Usage: $0 --cpu/gpu build_dir platform"
  echo ""
  echo "  --cpu/gpu      Specify CPU or GPU (compulsory)"
  echo "  build_dir      Specify the build directory"
  echo "  platform       Specify the platform"
  echo "  num_threads    Specify number of threads (used only for CPU)"
  echo ""
  echo "  Number of threads for CPU = 1 if unspecified"
  echo ""
  echo "Examples:"
  echo "  $0 --gpu build_sycl sycl"
  echo "  $0 --cpu build_genoa omp 8"
  echo "  $0 --cpu build_icelake threads"
  echo ""
  exit 0
}

if [[ $# -lt 3 ]]; then
  usage
else
  CPU_GPU="$1"
  BUILD_DIR="${SCRIPT_DIR}/$2"
  PLATFORM="$3"
  NUM_THREADS=$4
fi

if [ "$CPU_GPU" == "--gpu" ]; then
  VERSIONS=(${VERSIONS_GPU[*]})
elif [ "$CPU_GPU" == "--cpu" ]; then
  VERSIONS=(${VERSIONS_CPU[*]})
else
  echo "First option should be --gpu or --cpu"
  exit 1
fi

if [ ! -d "$BUILD_DIR" ]; then
  echo "$BUILD_DIR does not exist."
  exit 1
fi

OUT_DIR="out_${PLATFORM}"
for BENCH in "${BENCHMARKS[@]}"; do
  if [ ! -d "$OUT_DIR/${BENCH}_bench" ]; then
    mkdir -p "$OUT_DIR/${BENCH}_bench"
  fi
done

for BENCH in "${BENCHMARKS[@]}"; do
  for VERSION in "${VERSIONS[@]}"; do
    EXE="${BUILD_DIR}/build_${VERSION}/bench_mdrange_${BENCH}"
    OUT="${OUT_DIR}/${BENCH}_bench/${PLATFORM}_${BENCH}_${VERSION}.json"

    if [[ ! -x "${EXE}" ]]; then
      echo "Skipping ${BENCH}/${VERSION}: ${EXE} not built" >&2
      continue
    fi

    echo "==========================================="
    echo "Running ${BENCH} with version = ${VERSION}"
    echo "Output: ${OUT}"
    echo "==========================================="

    if [ "$CPU_GPU" == "--gpu" ]; then
        ${EXE} \
          --benchmark_out="${OUT}" \
          --benchmark_out_format=json
    else
        ${EXE} ${NUM_THREADS} \
          --benchmark_out="${OUT}" \
          --benchmark_out_format=json
    fi
  done
done
