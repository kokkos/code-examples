## To reproduce the half precision results for CGSolve:
```bash
git clone https://github.com/kokkos/kokkos.git
git clone https://github.com/kokkos/code-examples.git
cd code-examples
git checkout tags/us-rse-escience-2022
cd papers/kokkos-half-t-rse-escience-2022
```

### On V100
```bash
env KOKKOS_SRC_DIR=/path/to/kokkos ./run_benchmark_v100.sh
```

### On A100
```bash
env KOKKOS_SRC_DIR=/path/to/kokkos ./run_benchmark_a100.sh
```
