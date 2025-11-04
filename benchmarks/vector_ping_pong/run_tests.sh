#!/usr/bin/env bash

rm -rf build && cmake -B build -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_HOPPER90=ON -DKokkos_ARCH_NATIVE=ON -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_COMPILER_WARNINGS=ON && cmake --build build -j 10

nohup ./run.bash &
