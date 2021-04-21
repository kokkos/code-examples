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

#if !defined(USE_KOKKOS) && !defined(USE_OMPT) && !defined(USE_OMP)
#define USE_KOKKOS
#endif

template <typename ViewType, typename ValueType>
double min(ViewType a, ValueType &min_value) {
  Kokkos::Timer timer;

#if defined(USE_KOKKOS)
  Kokkos::parallel_reduce(
      "min", a.extent(0),
      KOKKOS_LAMBDA(int i, ValueType &lmin) {
        if (a(i) < lmin) lmin = a(i);
      },
      Kokkos::Min<ValueType>(min_value));
#else
  int const n = a.extent(0);
  auto const *a_ptr = a.data();
#if defined(USE_OMP)
  min_value = 1.e100;
  #pragma omp parallel for reduction(min: min_value)
  for(int i = 0; i < n; ++i)
    if (a(i) < min_value) min_value = a_ptr[i];
#elif defined(USE_OMPT)
  min_value = 1.e100;
#pragma omp target teams distribute parallel for reduction(min: min_value) is_device_ptr(a_ptr)
  for(int i = 0; i < n; ++i)
    if (a(i) < min_value) min_value = a_ptr[i];
#else
#error game over
#endif
#endif

  return timer.seconds();
}

template <typename ViewType, typename ValueType>
double sum(ViewType a, ValueType &sum_value) {
  Kokkos::Timer timer;

#if defined(USE_KOKKOS)
  Kokkos::parallel_reduce(
      "sum", a.extent(0),
      KOKKOS_LAMBDA(int i, ValueType &lsum) { lsum += a(i); },
      Kokkos::Sum<ValueType>(sum_value));
#else
  int const n = a.extent(0);
  auto const *a_ptr = a.data();
#if defined(USE_OMP)
  sum_value = 0.;
#pragma omp parallel for reduction(+: sum_value)
  for(int i = 0; i < n; ++i)
    sum_value += a_ptr[i];
#elif defined(USE_OMPT)
  sum_value = 0.;
#pragma omp target teams distribute parallel for reduction(+: sum_value) is_device_ptr(a_ptr)
  for(int i = 0; i < n; ++i)
    sum_value += a_ptr[i];
#else
#error game over
#endif
#endif

  return timer.seconds();
}

template <typename ViewType, typename ValueType>
double min_sum(ViewType a, ValueType &min_value, ValueType &sum_value) {
  Kokkos::Timer timer;
#if defined(USE_KOKKOS)
  Kokkos::parallel_reduce(
      "min_sum", a.extent(0),
      KOKKOS_LAMBDA(int i, ValueType &lmin, ValueType &lsum) {
        if (lmin > a(i)) lmin = a(i);
        lsum += a(i);
      },
      Kokkos::Min<ValueType>(min_value), Kokkos::Sum<ValueType>(sum_value));
#else
  int const n = a.extent(0);
  auto const *a_ptr = a.data();
#if defined(USE_OMP)
  sum_value = 0.;
  min_value = 1.e100;
#pragma omp parallel for reduction(+: sum_value) reduction(min: min_value)
  for(int i = 0; i < n; ++i) {
    if (a_ptr[i] < min_value) min_value = a_ptr[i];
    sum_value += a_ptr[i];
  }
#elif defined(USE_OMPT)
  sum_value = 0.
  min_value = 1e100;
#pragma omp target teams distribute parallel for reduction(+: sum_value) reduction(min: min_value) is_device_ptr(a_ptr)
  for(int i = 0; i < n; ++i) {
    if (a(i) < min_value) min_value = a_ptr[i];
    sum_value += a_ptr[i];
  }
#else
#error game over
#endif
#endif

  return timer.seconds();
}

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);

  int const size = 1000000;

  Kokkos::View<double *> a("a", size);

  Kokkos::parallel_for(
      "fill_view", size, KOKKOS_LAMBDA(int i) { a(i) = i + 1.; });
  Kokkos::fence();

  double min_value, sum_value;
  double duration_min, duration_sum;

  // Warmup
  duration_min = min(a, min_value);
  duration_min = min(a, min_value);
  printf("min value %lf, duration %e\n", min_value, duration_min);

  // Warmup
  duration_sum = sum(a, sum_value);
  duration_sum = sum(a, sum_value);
  printf("sum value %lf, duration %e\n", sum_value, duration_sum);

  double duration_min_sum;
  // Warmup
  duration_min_sum = min_sum(a, min_value, sum_value);
  duration_min_sum = min_sum(a, min_value, sum_value);
  printf("min value %lf, sum value %lf, duration %e\n", min_value, sum_value,
         duration_min_sum);

  return 0;
}
