#!/bin/bash
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/.versions.sh"
CPU_GPU=""
EXTRA_CMAKE_ARGS=()

usage() {
  echo "Usage: $0 [options] [-- cmake options]"
  echo ""
  echo "Show this help:"
  echo "  -h, --help"
  echo ""
  echo "Specify CPU or GPU backend (compulsory):"
  echo "  --cpu or --gpu"
  echo ""
  echo "Specify the build directory:"
  echo "  -B, --build-dir"
  echo ""
  echo "Extra CMake arguments can be passed after '--', e.g.:"
  echo "  -- -DKokkos_ENABLE_OPENMP=ON"
  echo ""
  echo "examples: "
  echo "  $0 --cpu"
  echo "  $0 --cpu --build-dir serial_build"
  echo "  $0 --cpu -- -DKokkos_ENABLE_THREADS=ON -DKokkos_ARCH_NATIVE=ON"
  echo "  $0 --gpu -B cuda_build -- -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON -DCMAKE_CXX_COMPILER=g++"
  echo "  $0 --gpu --build-dir hip_build -- -DKokkos_ENABLE_HIP=ON -DKokkos_ARCH_AMD_GFX90A=ON -DCMAKE_CXX_COMPILER=hipcc"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage ;;
    --cpu)
      CPU_GPU="cpu"
      shift ;;
    --gpu)
      CPU_GPU="gpu"
      shift ;;
    -B|--build-dir)
      BUILD_DIR="$2"
      shift 2
      ;;
    --)
      shift
      EXTRA_CMAKE_ARGS=("$@")
      break ;;
    *)
      echo "Unknown option: $1"; usage ;;
  esac
done

BUILD_FOLDER="${SCRIPT_DIR}/${BUILD_DIR:-build_dir}"

if [ -z "$CPU_GPU" ]; then
  echo "Either --cpu or --gpu is necessary!"
  exit 1
fi

if [ "$CPU_GPU" == "gpu" ]; then
  VERSIONS=(${VERSIONS_GPU[*]})
else
  VERSIONS=(${VERSIONS_CPU[*]})
fi

for VERSION in "${VERSIONS[@]}"; do
  BUILD_DIR="${BUILD_FOLDER}/build_${VERSION}"
  echo "========================================"
  echo "Configuring VERSION=${VERSION}"
  echo "Build dir: ${BUILD_DIR}"
  echo "========================================"

  cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=20 \
    -DVERSION="${VERSION}" \
    "${EXTRA_CMAKE_ARGS[@]}"
done
