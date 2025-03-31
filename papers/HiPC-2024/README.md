# APPS used in HiPC-2024 paper

Instructions to build benchmarks presented in `HiPC2024` paper titled "Leveraging LLVM OpenMP GPU Offload
Optimizations for Kokkos Applications"

CMake build is highly recommended.

Both the apps presented in the paper need Kokkos, so the first step is to install Kokkos with the corresponding backend.
We are interested in 2 backends per architecture, i.e., native and OpenMPTarget backend.


## Build Kokkos

The updates discussed in the paper to Kokkos are made to the `ompt_kernel_mode` branch of Kokkos fork [https://github.com/rgayatri23/kokkos.git](https://github.com/rgayatri23/kokkos.git)

All the extensions in LLVM/OpenMP discussed in the paper are available in the latest llvm release (as of writing this README) llvm/20.1.1. So please make sure that the relevant clang/clang++ compilers are installed and in your path.


### Native backends

Common CMake options across native and OpenMPTarget backends
`export COMMON_KK_OPTIONS="-DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$kokkos-backend-install -DKokkos_ARCH_$=ON -DKokkos_ENABLE_TESTS=OFF"`

> Disable unit tests in Kokkos to quickly build the library. Nothing wrong in building unit tests to test the install. However building unit tests with OpenMPTarget backend on AMD architecture takes multiple hours to build due to the `lto` issue so its advisable to avoid it.

Kokkos architecture options can be found [here](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html#architectures).

Some of the common ARCH options that are used in the paper and we will need soon are:

**For NVIDIA** \
Kokkos_ARCH_AMPERE80, KOKKOS_ARCH_HOPPER90

**For AMD** \
Kokkos_ARCH_AMD_GFX90A, Kokkos_ARCH_AMD_GFX942



***Native Backend***
```bash
cmake \
  -DCMAKE_CXX_COMPILER=nvcc/hipcc           \
  $COMMON_KK_OPTIONS                        \
  -DKokkos_ENABLE_CUDA/HIP=ON               \
  ../
```

***OpenMPTarget Backend***
```bash
cmake \
  -DCMAKE_CXX_COMPILER=clang++              \
  $COMMON_KK_OPTIONS                        \
  -DKokkos_ENABLE_OPENMPTARGET=ON           \
  ../
```

### Building the APPS

* AXPBY
```bash
cd axpby-dot/
cmake \
  -DCMAKE_CXX_COMPILER=$corresponding-backend-compiler       \
  -DCMAKE_BUILD_TYPE=Release                                 \
  -DKokkos_ROOT=$path-to-kokkos-install                      \
  ../
```

* CGSolve
```bash
cd cgsolve/
cmake \
  -DCMAKE_CXX_COMPILER=$corresponding-backend-compiler          \
  -DCMAKE_BUILD_TYPE=Release                                    \
  -DKokkos_ROOT=$path-to-kokkos-install                         \
  ../
```

* TestSNAP
```bash
cd TestSNAP/
cmake \
  -DCMAKE_CXX_COMPILER=$corresponding-backend-compiler          \
  -DCMAKE_BUILD_TYPE=Release                                    \
  -DKokkos_ROOT=$path-to-kokkos-install                         \
  -Dref_data=14                                                 \
  ../
```

## Makefile

Makefile is not the most tested build system but can be used if that is preferred.
Edit the first line of `Makefile.inc` to point to the Kokkos source.
The options set by default are for NVIDIA A100 architecture running with the OpenMPTarget backend of Kokkos.

```
