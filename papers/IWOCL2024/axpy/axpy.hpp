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
#include <cmath>

struct AXPBY {
    using view_t = Kokkos::View<double *>;
    int N;
    double alpha = 1.;
    double beta  = 2.;
    view_t x, y, z;

    bool fence_all;
    AXPBY(int N_, bool fence_all_)
        : N(N_),
          x(view_t("X", N)),
          y(view_t("Y", N)),
          z(view_t("Z", N)),
          fence_all(fence_all_) {}

    KOKKOS_FUNCTION
    void operator()(int i) const { z(i) = alpha*x(i) + beta*y(i); }

  double sycl_axpby(int R) {
    AXPBY f(*this);
    int N_ = N;
    sycl::queue q{sycl::property::queue::in_order()};
    auto x_data = x.data();
    auto y_data = y.data();
    auto z_data = z.data();
    double alpha = 1.;
    double beta  = 2.;
    // Warmup
                    q.parallel_for(sycl::range<1>(N_), [=](sycl::id<1> idx) {
                                    int i = idx;
				    z_data[i] = alpha*x_data[i] + beta*y_data[i];
				    });
    q.wait();

    Kokkos::Timer timer;
    for (int r = 0; r < R; r++) {
	                        q.parallel_for(sycl::range<1>(N_), [=](sycl::id<1> idx) {
                                     int i = idx;
				     z_data[i] = x_data[i] + y_data[i];
                                    });
    }
    q.wait();
    double time = timer.seconds();
    return time;
  }

    double kk_axpby(int R) {
        // Warmup
        Kokkos::parallel_for("kk_axpby_wup", N, *this);
        Kokkos::fence();

        Kokkos::Timer timer;
        for (int r = 0; r < R; r++) {
            Kokkos::parallel_for("kk_axpby", N, *this);
        }
        Kokkos::fence();
        double time = timer.seconds();
        return time;
    }

    void run_test(int R) {
        double bytes_moved = 1. * sizeof(double) * N * 3 * R;
        double GB = bytes_moved / 1024 / 1024 / 1024;

        double time_kk = kk_axpby(R);
	double time_sycl = sycl_axpby(R);
	std::cout << N << ":\t" << time_kk << " s\t" << GB/time_kk << " GB/s\t" << time_sycl << " s\t" << GB/time_sycl << " GB/s\t" << time_kk/time_sycl << '\n';
    }
};
