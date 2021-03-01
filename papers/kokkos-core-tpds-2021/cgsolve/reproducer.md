# CUDA build

```
# Kokkos config
cmake -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON -DCMAKE_INSTALL_PREFIX=${HOME}/Kokkos/install/3.3.01/Cuda/NVCC/11.1/kokkos -DKokkos_ARCH_VOLTA70=ON -DKokkos_ARCH_SKX=ON ~/Kokkos/kokkos

# KokkosKernels config
cmake -DKokkos_ROOT=/ascldap/users/crtrott/Kokkos/install/3.3.01/Cuda/NVCC/11.1/kokkos -DCMAKE_INSTALL_PREFIX=${
HOME}/Kokkos/install/3.3.01/Cuda/NVCC/11.1/kokkos-kernels -DCMAKE_CXX_COMPILER=${HOME}/Kokkos/install/3.3.01/Cuda/NVCC/11.1/kokkos/bin/nvcc_wrap
per -DKokkosKernels_ADD_DEFAULT_ETI=ON -DKokkosKernels_ENABLE_TPL_CUSPARSE=ON -DKokkosKernels_INST_DOUBLE=ON -DKokkosKernels_INST_MEMORYSPACE_CUDAUVMSPACE=OFF ~/Kokkos/kokkos-kernels

# cgsolve - native kernel config
cmake -DKokkos_ROOT=${HOME}/Kokkos/install/3.3.01/Cuda/NVCC/11.1/kokkos -DKokkosKernels_ROOT=${HOME}/Kokkos/install/3.3.01/Cuda/NVCC/11.1/kokkos-kernels -DCMAKE_CXX_COMPILER=${HOME}/Kokkos/install/3.3.01/Cuda/NVCC/11.1/kokkos/bin/nvcc_wrapper -DCMAKE_BUILD_TYPE=Release ../

# cgsolve - cusparse config
cmake -DKokkos_ROOT=${HOME}/Kokkos/install/3.3.01/Cuda/NVCC/11.1/kokkos -DKokkosKernels_ROOT=${HOME}/Kokkos/install/3.3.01/Cuda/NVCC/11.1/kokkos-kernels -DCMAKE_CXX_COMPILER=${HOME}/Kokkos/install/3.3.01/Cuda/NVCC/11.1/kokkos/bin/nvcc_wrapper -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-DUSE_KOKKOS_KERNELS" ../
```


# OpenMP Build:

```
# Kokkos config
cmake -DKokkos_ENABLE_OPENMP=ON -DCMAKE_INSTALL_PREFIX=${HOME}/Kokkos/install/3.3.01/OpenMP/Intel/19.5/kokkos -DKokkos_ARCH_VOLTA70=ON -DKokkos_ARCH_SKX=ON ~/Kokkos/kokkos

# KokkosKernels config
cmake -DKokkos_ROOT=/ascldap/users/crtrott/Kokkos/install/3.3.01/OpenMP/Intel/19.5/kokkos -DCMAKE_INSTALL_PREFIX=${HOME}/Kokkos/install/3.3.01/OpenMP/Intel/19.5/kokkos-kernels -DCMAKE_CXX_COMPILER=icpc -DCMAKE_BUILD_TYPE=Release -DKokkosKernels_ADD_DEFAULT_ETI=OFF -DKokkosKernels_ENABLE_TPL_MKL=ON ~/Kokkos/kokkos-kernels

# cgsolve - native kernel config
cmake -DKokkos_ROOT=${HOME}/Kokkos/install/3.3.01/OpenMP/Intel/19.5/kokkos -DKokkosKernels_ROOT=${HOME}/Kokkos/install/3.3.01/OpenMP/Intel/19.5/kokkos-kernels -DCMAKE_CXX_COMPILER=icpc -DCMAKE_BUILD_TYPE=Release ../


# cgsolve - mkl config
cmake -DKokkos_ROOT=${HOME}/Kokkos/install/3.3.01/OpenMP/Intel/19.5/kokkos -DKokkosKernels_ROOT=${HOME}/Kokkos/install/3.3.01/OpenMP/Intel/19.5/kokkos-kernels -DCMAKE_CXX_COMPILER=icpc -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-DUSE_MKL" ../
```
