#!/usr/bin/env bash

rm -rf build_h100 && cmake -B build_h100 -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_HOPPER90=ON -DKokkos_ARCH_NATIVE=ON -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_COMPILER_WARNINGS=ON && cmake --build build_h100 -j 10

echo "OMP_PROC_BIND=spread OMP_PLACES=threads ./run.bash &> run_h100.out" | at 22:00
echo "Queued H100 job"
