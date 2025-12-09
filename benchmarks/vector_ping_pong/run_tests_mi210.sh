#!/usr/bin/env bash

rm -rf build_mi210 && cmake -B build_mi210 -DKokkos_ENABLE_HIP=ON -DKokkos_ARCH_AMD_GFX90A=ON -DKokkos_ARCH_NATIVE=ON -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_COMPILER_WARNINGS=ON -DCMAKE_CXX_COMPILER=hipcc && cmake --build build_mi210 -j 10

echo "OMP_PROC_BIND=spread OMP_PLACES=threads HSA_XNACK=1 ./run.bash ./build_mi210/vector_ping_pong mi210 &> run_mi210.out" | at 22:00
echo "Queued MI210 job"
