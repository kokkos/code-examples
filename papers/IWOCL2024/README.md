# Code examples for paper on SYCL backend of Kokkos - IWOCL 2024

## Build 

Build Kokkos (commit 18d7d78f5df94694fd37348051b20d7e6db3b1ac was used) with the `OpenMP` and `SYCL` backend enabled:
```shell
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER =icpx \
  -DKokkos_ARCH_INTEL_PVC=ON \
  -DKokkos_ARCH_SPR=ON \
  -DKokkos_ENABLE_OPENMP=ON \
  -DKokkos_ENABLE_SYCL=ON \
 ..
```
The compiler used comes from the `oneapi/eng-compiler/2023.10.15.002` module and identifies as
```
Intel(R) oneAPI DPC++/C++ Compiler 2024.0.0 (2024.x.0.20231009)
```  
For running with `SYCL+OpenCL` replace `Kokkos_ARCH_INTEL_PVC` with `Kokkos_ARCH_INTEL_GEN`.
This also requires applying
```diff
diff --git a/core/src/SYCL/Kokkos_SYCL.cpp b/core/src/SYCL/Kokkos_SYCL.cpp
index af64b6908..cdda12020 100644
--- a/core/src/SYCL/Kokkos_SYCL.cpp
+++ b/core/src/SYCL/Kokkos_SYCL.cpp
@@ -147,13 +147,13 @@ void SYCL::impl_initialize(InitializationSettings const& settings) {
   // If the device id is not specified and there are no GPUs, sidestep Kokkos
   // device selection and use whatever is available (if no GPU architecture is
   // specified).
-#if !defined(KOKKOS_ARCH_INTEL_GPU) && !defined(KOKKOS_IMPL_ARCH_NVIDIA_GPU)
+//#if !defined(KOKKOS_ARCH_INTEL_GPU) && !defined(KOKKOS_IMPL_ARCH_NVIDIA_GPU)
   if (!settings.has_device_id() && gpu_devices.empty()) {
     Impl::SYCLInternal::singleton().initialize(sycl::device());
     Impl::SYCLInternal::m_syclDev = 0;
     return;
   }
-#endif
+//#endif
   const auto id = ::Kokkos::Impl::get_gpu(settings);
   Impl::SYCLInternal::singleton().initialize(gpu_devices[id]);
   Impl::SYCLInternal::m_syclDev = id;
```
Then, using `OpenCL` is forced via
```bash
export ONEAPI_DEVICE_SELECTOR=opencl:cpu
```


### AXPBY

Simply compile the bechmark in `Release` mode linking to the Kokkos installation (`-DKokkos_ROOT=...`) and run the executable.

### DOT

Simply compile the bechmark in `Release` mode linking to the Kokkos installation (`-DKokkos_ROOT=...`) and run the executable.

### CGSOLVE

Using `oneMKL` requires `KokkosKernels` (commit 3dafbed7bb4f76d3bc38c923297d9455d9e7849e was used) built with `-DKokkosKernels_ENABLE_TPL_MKL:BOOL=ON`.
The `oneMKL` library identifies as `oneMKL 2024.0.1`.
To be enabl to use `oneMKL` in the benchmark, configure with `-DUSE_MKL=ON`. 

The benchmark has two configuration macros in `cgsolve.hpp`:
- USE_TPL: Determines if `oneMKL` is used for the `Kokkos` results
- ONLY_DO_SPMV: Determines if only the SPMV part should be run or the full CG algorithm. 

Simply compile the bechmark in `Release` mode linking to the Kokkos installation (`-DKokkos_ROOT=...`) 
and possibly to the KokkosKernels installation (`=DKokkosKernels_ROOT=...`) and run the executable.
