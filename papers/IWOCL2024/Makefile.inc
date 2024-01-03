KOKKOS_PATH=$(SCRATCH)/Kokkos/kokkos-4.0.01
SRC = $(wildcard *.cpp)
KOKKOS_CXX_STANDARD=c++17

OFFLOAD_FLAGS ?= -fopenmp
arch ?= A100

ifeq ($(arch), A100)
    KOKKOS_ARCH=Ampere80
else ifeq ($(arch), MI250x)
    KOKKOS_ARCH=Vega90A
else ifeq ($(arch), PVC)
    KOKKOS_ARCH=INTEL_PVC
endif

ifeq ($(backend), cuda)
    ifeq ($(comp), clang)
        CXX = clang++
        KOKKOS_CUDA_OPTIONS += "enable_lambda"
        KOKKOS_DEVICES = "Cuda"
    else
        CXX = $(KOKKOS_PATH)/bin/nvcc_wrapper
        KOKKOS_CUDA_OPTIONS += "enable_lambda"
        KOKKOS_DEVICES = "Cuda"
    endif
else ifeq ($(backend), hip)
    CXX = hipcc
    KOKKOS_DEVICES = "HIP"
else ifeq ($(backend), sycl)
    CXX = icpx
    KOKKOS_DEVICES = "SYCL"
else ifeq ($(backend), ompt)
    KOKKOS_DEVICES = "OpenMPTarget"
    OFFLOAD_FLAGS = -fopenmp
    ifeq ($(comp), clang)
        CXX = clang++
        ifeq ($(arch), A100)
            OFFLOAD_FLAGS += -Wno-unknown-cuda-version -Wno-undefined-internal -Wno-pass-failed -Wno-openmp-mapping -Wno-extra-tokens
        else ifeq ($(arch), MI250x)
            OFFLOAD_FLAGS += --offload-arch=gfx90a
        endif
    else ifeq ($(comp), nvhpc)
        KOKKOS_OPTIONS += KOKKOS_INTERNAL_DISABLE_DESUL_ATOMICS
        CXX = nvc++
        OFFLOAD_FLAGS = -mp=gpu -gpu=cc80
    else ifeq ($(comp), cce)
        CXX = CC
        OFFLOAD_FLAGS =
    else ifeq ($(comp), rocmclang)
        CXX = amdclang++
        OFFLOAD_FLAGS += -target x86_64-pc-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a
    else ifeq ($(comp), icpx)
        CXX = icpx 
	OFFLOAD_FLAGS += -Xclang -fopenmp-use-host-usm-for-implicit-reduction-map
    endif
endif
