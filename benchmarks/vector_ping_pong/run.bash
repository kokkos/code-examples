#!/usr/bin/env bash

declare -a variations=("device-hostpinned" "device-new" "device-managed" "managed-none" "new-none" "malloc-none" "device-none")

for warmups in 100
do
  for pingpongs in 100
  do
    for stride in 1 2 4 8 16 32
    do
      for size in 2**10 2**12 2**14 2**16 2**20 2**22
      do
         for variation in "${variations[@]}"
         do
            ./Kokkos_vector_ping_pong "$variation" 10 "$((size))" "${warmups}" "${pingpongs}" "${stride}"
         done
       done
    done
  done
done
