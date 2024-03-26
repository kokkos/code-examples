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

/*
  Adapted from the Mantevo Miniapp Suite.
  https://mantevo.github.io/pdfs/MantevoOverview.pdf
*/

#include "generate_matrix.hpp"
//#include "cgsolve.hpp"
#include "cgsolve_ompx.hpp"

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc,argv);
  {
    int N = argc>1?atoi(argv[1]):100;
    int max_iter = argc>2?atoi(argv[2]):200;
    double tolerance = argc>3?atoi(argv[3]):1e-7;

    // Case 1
    {
    cgsolve obj(150, max_iter, tolerance);
    obj.run_test();
    }

    // Case 2
    {
    cgsolve obj(255, max_iter, tolerance);
    obj.run_test();
    }

    // Case 3
    {
    cgsolve obj(325, max_iter, tolerance);
    obj.run_test();
    }

  }
  Kokkos::finalize();

  return 0;
}
