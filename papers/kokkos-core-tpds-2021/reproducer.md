# Kokkos Core Build Instructions

Kokkos is build from version 3.3.01. We are using:
* CUDA 11.1
* GCC 9.2.0
* Intel 19.0.5

KokkosKernels is only needed for the CGSolve example to run the TPL Comparison. Version 3.3.01 is used

## Clone repositories
```
git clone https://github.com/kokkos/kokkos
git clone https://github.com/kokkos/kokkos-kernels
cd kokkos
git checkout 3.3.01
cd ../kokkos-kernels
git checkout 3.3.01
```

## CUDA Configuration of Kokkos and KokkosKernels

```
# Kokkos config
cmake -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON -DCMAKE_INSTALL_PREFIX=${HOME}/Kokkos/install/3.3.01/Cuda/NVCC/11.1/kokkos -DKokkos_ARCH_VOLTA70=ON -DKokkos_ARCH_SKX=ON ~/Kokkos/kokkos

# KokkosKernels config
cmake -DKokkos_ROOT=/ascldap/users/crtrott/Kokkos/install/3.3.01/Cuda/NVCC/11.1/kokkos -DCMAKE_INSTALL_PREFIX=${
HOME}/Kokkos/install/3.3.01/Cuda/NVCC/11.1/kokkos-kernels -DCMAKE_CXX_COMPILER=${HOME}/Kokkos/install/3.3.01/Cuda/NVCC/11.1/kokkos/bin/nvcc_wrap
per -DKokkosKernels_ADD_DEFAULT_ETI=ON -DKokkosKernels_ENABLE_TPL_CUSPARSE=ON -DKokkosKernels_INST_DOUBLE=ON -DKokkosKernels_INST_MEMORYSPACE_CUDAUVMSPACE=OFF ~/Kokkos/kokkos-kernels
```


## OpenMP Configuration of Kokkos and KokkosKernels:

```
# Kokkos config
cmake -DKokkos_ENABLE_OPENMP=ON -DCMAKE_INSTALL_PREFIX=${HOME}/Kokkos/install/3.3.01/OpenMP/Intel/19.5/kokkos -DKokkos_ARCH_VOLTA70=ON -DKokkos_ARCH_SKX=ON ~/Kokkos/kokkos

# KokkosKernels config
cmake -DKokkos_ROOT=/ascldap/users/crtrott/Kokkos/install/3.3.01/OpenMP/Intel/19.5/kokkos -DCMAKE_INSTALL_PREFIX=${HOME}/Kokkos/install/3.3.01/OpenMP/Intel/19.5/kokkos-kernels -DCMAKE_CXX_COMPILER=icpc -DCMAKE_BUILD_TYPE=Release -DKokkosKernels_ADD_DEFAULT_ETI=OFF -DKokkosKernels_ENABLE_TPL_MKL=ON ~/Kokkos/kokkos-kernels
```
