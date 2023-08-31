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
#include <axpby.hpp>
#include <dot.hpp>
#include <cmath>

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    int N = argc > 1 ? atoi(argv[1]) : 1000000;
    int R = argc > 2 ? atoi(argv[2]) : 10;

    printf("***********AXPBY******************\n");
    // Case 1 - 2GBs
    {
      AXPBY axpby(10000000, false);
      axpby.run_test(R);
    }

    // Case 2 - 11GBs
    {
      AXPBY axpby(50000000, false);
      axpby.run_test(R);
    }

    // Case 3 - 22GBs
    {
      AXPBY axpby(100000000, false);
      axpby.run_test(R);
    }
    

    printf("\n***********DOT******************\n");
    R = 15;

    // Case 1 - 2GBs
    {
        DOT dot(10000000, false);
        dot.run_test(R);
    }

    // Case 2 - 11GBs
    {
        DOT dot(50000000, false);
        dot.run_test(R);
    }

    // Case 3 - 12GBs
    {
        DOT dot(100000000, false);
        dot.run_test(R);
    }
  }
  Kokkos::finalize();
}
