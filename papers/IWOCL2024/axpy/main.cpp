//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <Kokkos_Core.hpp>
#include <axpy.hpp>
#include <cmath>

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    int N = argc > 1 ? atoi(argv[1]) : INT_MAX;
	                             //2'147'483'647;
    int R = argc > 2 ? atoi(argv[2]) : 500;
    for (unsigned int n = 2; n<INT_MAX; n*=2) {
      AXPBY axpby(n, false);
      axpby.run_test(R);
    }
  }
  Kokkos::finalize();
}
