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

template <typename TensorType>
double tensor_add_only_first_dimension(TensorType a, TensorType b) {
  int const size0 = a.extent(0);
  int const size1 = a.extent(1);
  int const size2 = a.extent(2);
  Kokkos::Timer timer;
  Kokkos::parallel_for(
      "tensor_add_only_first_dimension", Kokkos::RangePolicy<>(0, size0),
      KOKKOS_LAMBDA(int i) {
        for (int j = 0; j < size1; ++j)
          for (int k = 0; k < size2; ++k) {
            a(i, j, k) += b(i, j, k);
          }
      });
  Kokkos::fence();

  return timer.seconds();
}

template <typename TensorType>
double tensor_add_flattened(TensorType a, TensorType b) {
  int const size0 = a.extent(0);
  int const size1 = a.extent(1);
  int const size2 = a.extent(2);
  Kokkos::Timer timer;
  Kokkos::parallel_for(
      "tensor_add_flattened", Kokkos::RangePolicy<>(0, size0 * size1 * size2),
      KOKKOS_LAMBDA(int n) {
        int const i = n / (size1 * size2);
        int const j = (n % (size1 * size2)) / size2;
        int const k = n % size2;
        a(i, j, k) += b(i, j, k);
      });
  Kokkos::fence();

  return timer.seconds();
}

template <typename TensorType>
double tensor_add_mdrange(TensorType a, TensorType b) {
  int const size0 = a.extent(0);
  int const size1 = a.extent(1);
  int const size2 = a.extent(2);
  Kokkos::Timer timer;
  Kokkos::parallel_for(
      "tensor_add_mdrange",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({{0, 0, 0}},
                                             {{size0, size1, size2}}),
      KOKKOS_LAMBDA(int i, int j, int k) { a(i, j, k) += b(i, j, k); });
  Kokkos::fence();

  return timer.seconds();
}

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);

  int const size = 100;

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
    const double duration = tensor_add_only_first_dimension(A, B);
    std::cout << "tensor_add_only_first_dimension duration(s) " << duration
              << '\n';
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
    const double duration = tensor_add_flattened(A, B);
    std::cout << "tensor_add_flattened duration(s) " << duration << '\n';
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
    const double duration = tensor_add_mdrange(A, B);
    std::cout << "tensor_add_mdrange duration(s) " << duration << '\n';
  }

  return 0;
}
