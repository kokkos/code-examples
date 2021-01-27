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

#include <numeric>

template <typename ViewType>
typename ViewType::value_type dot_product(ViewType a, ViewType b)
{
   using ValueType = typename ViewType::value_type;
   ValueType result;
   Kokkos::parallel_reduce("dot_product", a.extent(0), KOKKOS_LAMBDA(int i, ValueType &lsum)
                        {
                          lsum += a(i)*b(i);
                        }, Kokkos::Sum<ValueType>(result));

   return result;
}

int main(int argc, char* argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  int const size = 1000;

  Kokkos::View<double*> a("a", size);
  Kokkos::View<double*> b("b", size);

  Kokkos::parallel_for("fill_view", size, KOKKOS_LAMBDA(int i)
                       {
                         a(i) = i;
                         b(i) = i+1;
                       });
  double const result = dot_product(a, b);
  
  std::vector<double> a_ref(size);
  std::iota(a_ref.begin(), a_ref.end(), 0);
  std::vector<double> b_ref(size);
  std::iota(b_ref.begin(), b_ref.end(), 1);
  double const reference = std::inner_product(a_ref.begin(), a_ref.end(), b_ref.begin(), 0.);

  printf("dot_product %f, reference value %f\n", result, reference);

  return 0;
}
