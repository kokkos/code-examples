#!/bin/bash
export KOKKOS_SRC_DIR=${KOKKOS_SRC_DIR:-"$HOME/KOKKOS.base/kokkos"}
export KOKKOS_SRC_DIR=$(realpath $KOKKOS_SRC_DIR)
export KOKKOS_SHA=${KOKKOS_SHA:-"tags/3.6.00"}
module purge
module load sems-archive-env sems-env sems-gcc/8.3.0 sems-cmake/3.19.1 cuda/11.2 sems-archive-git/2.10.1
cd $KOKKOS_SRC_DIR
git checkout $KOKKOS_SHA
cd -

config="dddd"
make -f Makefile.$config
./run_size_sweep $config 10 20

config="ffff"
make -f Makefile.$config
./run_size_sweep $config 10 20

config="hhff"
make -f Makefile.$config
./run_size_sweep $config 10 20

config="hhfh"
make -f Makefile.$config
./run_size_sweep $config 10 20

./analyse_all
