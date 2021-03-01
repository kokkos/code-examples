# CUDA build

```
cmake -DKokkos_ROOT=${HOME}/Kokkos/install/3.3.01/Cuda/NVCC/11.1/kokkos -DCMAKE_CXX_COMPILER=${HOME}/Kokkos/install/3.3.01/Cuda/NVCC/11.1/kokkos/bin/nvcc_wrapper -DCMAKE_BUILD_TYPE=Release ../
```


# OpenMP Build:

```
cmake -DKokkos_ROOT=${HOME}/Kokkos/install/3.3.01/OpenMP/Intel/19.5/kokkos -DCMAKE_CXX_COMPILER=icpc -DCMAKE_BUILD_TYPE=Release ../
```

# Benchmark Run
```
./run_bench atomic 20
```

This produces atomic.raw_numbers.cvs which was used to compute averages and errors.
