To reproduce the half precision results for CGSolve:
```bash
git clone https://github.com/kokkos/kokkos.git
git clone https://github.com/kokkos/code-examples.git
cd code-examples
git checkout half-precision
cd papers/kokkos-half-t-rse-escience-2022
env KOKKOS_SRC_DIR=/path/to/kokkos ./run_benchmark.sh
```