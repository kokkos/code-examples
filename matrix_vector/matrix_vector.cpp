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

template <typename MatrixType, typename ViewType>
double matrix_vector(MatrixType A,ViewType a, ViewType b)
{
   int const size = a.extent(0);
   Kokkos::Timer timer;
   Kokkos::parallel_for("matrix_vector", size, KOKKOS_LAMBDA(int i)
                        {
                          for (int j=0; j<size; ++j)
                          {
                            b(i) += A(i,j) * a(j);
                          }
                        });
  Kokkos::fence();

  return timer.seconds();
}

int main(int argc, char* argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  int const size = 10000;
  Kokkos::View<double*> a("a", size);
  Kokkos::View<double*> b("b", size);
  Kokkos::parallel_for("fill_a", size, KOKKOS_LAMBDA(int i)
                       {
                         a(i) = i;
                       });

  double layout_left_duration = 0.;
  double layout_right_duration = 0.;

  // Layout left
  {
    Kokkos::View<double**, Kokkos::LayoutLeft> A("A", size, size);
    Kokkos::parallel_for("fill_A_left", size*size, KOKKOS_LAMBDA(int i)
                         {
                           A(i%size, i/size) = i;
                         });
    Kokkos::fence();
    layout_left_duration = matrix_vector(A, a, b);
  }

  printf("layout left duration(s) %f\n", layout_left_duration);

  // Layout right
  {
    Kokkos::View<double**, Kokkos::LayoutRight> A("A", size, size);
    Kokkos::parallel_for("fill_A_right", size*size, KOKKOS_LAMBDA(int i)
                         {
                           A(i%size, i/size) = i;
                         });
    Kokkos::fence();
    layout_right_duration = matrix_vector(A, a, b);
  }

  printf("layout right duration(s) %f\n", layout_right_duration);

  return 0;
}
