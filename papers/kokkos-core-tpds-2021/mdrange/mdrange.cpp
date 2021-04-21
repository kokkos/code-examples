/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <Kokkos_Core.hpp>

#if !defined(USE_KOKKOS) && !defined(USE_OMPT) && !defined(USE_OMP) && !defined(USE_CUDA)
#define USE_KOKKOS
#endif

#if defined(USE_CUDA)
template<class LAMBDA>
__global__ void cuda_kernel(size_t N, const LAMBDA f) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<N) f(i);
}
template<class LAMBDA>
__global__ void cuda_kernel3(size_t N0, size_t N1, size_t N2, const LAMBDA f) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if(i<N0 && j<N1 && k<N2) f(i,j,k);
}
#endif

template <typename TensorType>
double tensor_add_only_first_dimension(TensorType a, TensorType b) {
  int const size0 = a.extent(0);
  int const size1 = a.extent(1);
  int const size2 = a.extent(2);
  Kokkos::Timer timer;
#if defined(USE_KOKKOS)
  Kokkos::parallel_for(
      "tensor_add_only_first_dimension", Kokkos::RangePolicy<>(0, size0),
      KOKKOS_LAMBDA(int i) {
        for (int j = 0; j < size1; ++j)
          for (int k = 0; k < size2; ++k) {
            a(i, j, k) += b(i, j, k);
          }
      });
  Kokkos::fence();
#else
  auto * a_ptr = a.data();
  auto const * b_ptr = b.data();
#if defined(USE_OMP)
  #pragma omp parallel for
  for (int i = 0; i < size0; ++i)
    for (int j = 0; j < size1; ++j)
      for (int k = 0; k < size2; ++k) {
        int const idx = (i * size1 + j ) * size2 + k;
        a_ptr[idx] += b_ptr[idx];
      }
#elif defined(USE_CUDA)
  cuda_kernel<<<(size0+128)/128,128>>>(size0,[=]__device__(int i) {
        for (int j = 0; j < size1; ++j)
          for (int k = 0; k < size2; ++k) {
            int const idx = i + j * size0 + k * size0 * size1;
            a_ptr[idx] += b_ptr[idx];
          }
  });
  Kokkos::fence();
#elif defined(USE_OMPT)
  #pragma omp target teams distribute parallel for is_device_ptr(a_ptr, b_ptr)
  for (int i = 0; i < size0; ++i)
    for (int j = 0; j < size1; ++j)
      for (int k = 0; k < size2; ++k) {
        int const idx = i + j * size0 + k * size0 * size1;
        a_ptr[idx] += b_ptr[idx];
      }
#else
#error logic error
#endif
#endif

  return timer.seconds();
}

template <typename TensorType>
double tensor_add_flattened(TensorType a, TensorType b) {
  int const size0 = a.extent(0);
  int const size1 = a.extent(1);
  int const size2 = a.extent(2);
  Kokkos::Timer timer;
#if defined(USE_KOKKOS)
  Kokkos::parallel_for(
      "tensor_add_flattened", Kokkos::RangePolicy<>(0, size0 * size1 * size2),
      KOKKOS_LAMBDA(int n) {
        int const i = n / (size1 * size2);
        int const j = (n % (size1 * size2)) / size2;
        int const k = n % size2;
        a(i, j, k) += b(i, j, k);
      });
  Kokkos::fence();
#else
  auto * a_ptr = a.data();
  auto const * b_ptr = b.data();
#if defined(USE_OMP)
#pragma omp parallel for
  for (int n = 0; n < size0 * size1 * size2; ++n) {
    int const i = n / (size1 * size2);
    int const j = (n % (size1 * size2)) / size2;
    int const k = n % size2;
    int const idx = (i * size1 + j ) * size2 + k;
    a_ptr[idx] += b_ptr[idx];
  }
#elif defined(USE_CUDA)
  cuda_kernel<<<(size0*size1*size2+127)/128,128>>>(size0*size1*size2,[=]__device__(int n) {
        int const i = n / (size1 * size2);
        int const j = (n % (size1 * size2)) / size2;
        int const k = n % size2;
        int const idx = i + j * size0 + k * size0 * size1;
        a_ptr[idx] += b_ptr[idx];
  });
  Kokkos::fence();
#elif defined(USE_OMPT)
#pragma omp target teams distribute parallel for is_device_ptr(a_ptr, b_ptr)
  for (int n = 0; n < size0 * size1 * size2; ++n) {
    int const i = n / (size1 * size2);
    int const j = (n % (size1 * size2)) / size2;
    int const k = n % size2;
    int const idx = i + j * size0 + k * size0 * size1;
    a_ptr[idx] += b_ptr[idx];
  }
#else
#error logic error
#endif
#endif
  return timer.seconds();
}

template <typename TensorType>
double tensor_add_mdrange(TensorType a, TensorType b) {
  int const size0 = a.extent(0);
  int const size1 = a.extent(1);
  int const size2 = a.extent(2);
  Kokkos::Timer timer;
#if defined(USE_KOKKOS)
  Kokkos::parallel_for(
      "tensor_add_mdrange",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({{0, 0, 0}},
                                             {{size0, size1, size2}}),
      KOKKOS_LAMBDA(int i, int j, int k) { a(i, j, k) += b(i, j, k); });
  Kokkos::fence();
#else
  auto* a_ptr       = a.data();
  auto const* b_ptr = b.data();
#if defined(USE_OMP)
#pragma omp parallel for collapse(3)
  for (int i = 0; i < size0; ++i) {
    for (int j = 0; j < size1; ++j) {
      for (int k = 0; k < size2; ++k) {
        int const idx = (i * size1 + j ) * size2 + k;
        a_ptr[idx] += b_ptr[idx];
      }
    }
  }
#elif defined(USE_CUDA)
  dim3 block = {16,4,4};
  dim3 grid = {(size0+15)/16, (size1+3)/4, (size2+3)/4};
  cuda_kernel3<<<grid,block>>>(size0,size1,size2,[=]__device__(int i, int j, int k) {
    int const idx = i + j * size0 + k * size0 * size1;
    a_ptr[idx] += b_ptr[idx];
  });
  Kokkos::fence();

#elif defined(USE_OMPT)
#pragma omp target teams distribute parallel for collapse(3) is_device_ptr(a_ptr, b_ptr)
  for (int k = 0; k < size2; ++k) {
    for (int j = 0; j < size1; ++j) {
      for (int i = 0; i < size0; ++i) {
        int const idx = i + j * size0 + k * size0 * size1;
        a_ptr[idx] += b_ptr[idx];
      }
    }
  }
#else
#error logic error
#endif
#endif
  return timer.seconds();
}

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);

  int const size = 200;

  double GB_moved = 1. * size * size * size * 3 * 8 / 1024 / 1024 / 1024;
  // tensor_add_only_first_dimension
  {
    Kokkos::View<double***> A("A", size, size, size);
    Kokkos::View<double***> B("B", size, size, size);
    Kokkos::parallel_for(
        "fill",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({{0, 0, 0}},
                                               {{size, size, size}}),
        KOKKOS_LAMBDA(int i, int j, int k) {
          A(i, j, k) = i * j * k;
          B(i, j, k) = i + j + k;
        });
    Kokkos::fence();
    // Warmup
    double duration = tensor_add_only_first_dimension(A, B);
    duration        = tensor_add_only_first_dimension(A, B);
    printf("tensor_add_only_first_dimension duration(s) %lfus %lfGB/s\n",
           duration * 1.e6, GB_moved / duration);
  }

  // tensor_add_flattened
  {
    Kokkos::View<double***> A("A", size, size, size);
    Kokkos::View<double***> B("B", size, size, size);
    Kokkos::parallel_for(
        "fill",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({{0, 0, 0}},
                                               {{size, size, size}}),
        KOKKOS_LAMBDA(int i, int j, int k) {
          A(i, j, k) = i * j * k;
          B(i, j, k) = i + j + k;
        });
    Kokkos::fence();
    // Warmup
    double duration = tensor_add_flattened(A, B);
    duration        = tensor_add_flattened(A, B);
    printf("tensor_add_flattened duration(s) %lfus %lfGB/s\n", duration * 1.e6,
           GB_moved / duration);
  }

  // tensor_add_mdrange
  {
    Kokkos::View<double***> A("A", size, size, size);
    Kokkos::View<double***> B("B", size, size, size);
    Kokkos::parallel_for(
        "fill",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({{0, 0, 0}},
                                               {{size, size, size}}),
        KOKKOS_LAMBDA(int i, int j, int k) {
          A(i, j, k) = i * j * k;
          B(i, j, k) = i + j + k;
        });
    Kokkos::fence();
    // Warmup
    double duration = tensor_add_mdrange(A, B);
    duration        = tensor_add_mdrange(A, B);
    printf("tensor_add_mdrange duration(s) %lfus %lfGB/s\n", duration * 1.e6,
           GB_moved / duration);
  }

  return 0;
}
