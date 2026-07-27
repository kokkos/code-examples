#!/bin/bash
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/.versions.sh"
BUILD_FOLDER=""
PLATFORM=""

usage() {
  echo "Usage: $0 build_dir platform"
  echo ""
  echo "  build_dir      Specify the build directory"
  echo "  platform       Specify the platform"
  echo ""
  echo "Example:"
  echo "  $0 build_sycl sycl"
  echo ""
  exit 0
}

if [[ $# -lt 2 ]]; then
  usage
else
  BUILD_FOLDER="${SCRIPT_DIR}/$1"
  PLATFORM="$2"
fi

if [ ! -d "$BUILD_FOLDER" ]; then
  echo "$BUILD_FOLDER does not exist."
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
    EXE="${BUILD_FOLDER}/build_${VERSION}/bench_mdrange_${BENCH}"
    OUT="${OUT_DIR}/${BENCH}_bench/${PLATFORM}_${BENCH}_${VERSION}.json"
 
    if [[ ! -x "${EXE}" ]]; then
      echo "Skipping ${BENCH}/${VERSION}: ${EXE} not built" >&2
      continue
    fi
 
    echo "==========================================="
    echo "Running ${BENCH} with version = ${VERSION}"
    echo "Output: ${OUT}"
    echo "==========================================="
 
    ${EXE} \
      --benchmark_out="${OUT}" \
      --benchmark_out_format=json
  done
done
