#!/bin/bash
set -e

VERSIONS=("502" "device_iterate" "tile" "no_stride")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRA_CMAKE_ARGS=()

usage() {
  echo "Usage: $0 [options] [-- cmake options]"
  echo ""
  echo "  -h, --help         Show this help"
  echo "  -B, --build-dir    Specify the build directory"
  echo ""
  echo "Extra CMake arguments can be passed after '--', e.g.:"
  echo "  $0 -- -DKokkos_ENABLE_OPENMP=ON"
  echo ""
  echo "examples: "
  echo "  $0 --build-dir cuda_build -- -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON -DCMAKE_CXX_COMPILER=g++"
  echo "  $0 --build-dir hip_build -- -DKokkos_ENABLE_HIP=ON --DKokkos_ARCH_AMD_GFX90A=ON -DCMAKE_CXX_COMPILER=hipcc"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage ;;
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

