#!/bin/bash
export KOKKOS_SRC_DIR=${KOKKOS_SRC_DIR:-"$HOME/KOKKOS.base/kokkos"}
export KOKKOS_SRC_DIR=$(realpath $KOKKOS_SRC_DIR)
export KOKKOS_SHA=${KOKKOS_SHA:-"tags/3.6.00"}
export PROFILE_CMD="nsys profile --stats=true -t nvtx,cuda"
module purge
module load cudatoolkit/11.2 cmake/3.22.0
cd $KOKKOS_SRC_DIR
git checkout $KOKKOS_SHA
cd -

config="dddd"
sed -i 's/SKX,Volta70/AMPERE80/g' Makefile.$config
make -f Makefile.$config
./run_size_sweep $config 10 20

config="ffff"
sed -i 's/SKX,Volta70/AMPERE80/g' Makefile.$config
make -f Makefile.$config
./run_size_sweep $config 10 20

config="hhff"
sed -i 's/SKX,Volta70/AMPERE80/g' Makefile.$config
make -f Makefile.$config
./run_size_sweep $config 10 20

config="hhfh"
sed -i 's/SKX,Volta70/AMPERE80/g' Makefile.$config
make -f Makefile.$config
./run_size_sweep $config 10 20

./analyse_all
