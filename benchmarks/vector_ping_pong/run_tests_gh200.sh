#!/usr/bin/env bash

module load cmake/3.31.1 gcc/12.2.0 cuda/13.0
rm -rf build_gh200 && cmake -B build_gh200 -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_HOPPER90=ON -DKokkos_ARCH_NATIVE=ON -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_COMPILER_WARNINGS=ON && cmake --build build_gh200 -j 10

echo "OMP_PROC_BIND=spread OMP_PLACES=threads ./run.bash ./build_gh200/vector_ping_pong gh200 &> run_gh200.out" | at 22:00
echo "Queued GH200 job"
