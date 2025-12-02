#!/usr/bin/env bash

rm -rf build_mi300a && cmake -B build_mi300a -DKokkos_ENABLE_HIP=ON -DKokkos_ARCH_AMD_GFX942_APU=ON -DKokkos_ARCH_NATIVE=ON -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_COMPILER_WARNINGS=ON -DCMAKE_CXX_COMPILER=hipcc && cmake --build build_mi300a -j 10

echo "OMP_PROC_BIND=spread OMP_PLACES=threads ./run.bash &> run_mi300a.out" | at 22:00
echo "Queued MI300A job"
