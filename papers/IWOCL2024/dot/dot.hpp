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
#include <iostream>

struct DOT {
  using view_t = Kokkos::View<double *>;
  int N;
  view_t x, y, z;
  double* x_data;
  double* y_data;

  bool fence_all;
  DOT(int N_, bool fence_all_)
      : N(N_), x(view_t("X", N)), y(view_t("Y", N)), x_data(x.data()), y_data(y.data()), fence_all(fence_all_) {}

  KOKKOS_FUNCTION
  void operator()(int i, double &lsum) const { lsum += x(i) * y(i); }

  double kk_dot(int R) {
    // Warmup
    double result;
    auto x_data_ = x_data;
    auto y_data_ = y_data;
    Kokkos::parallel_reduce("kk_dot_wup", N, KOKKOS_LAMBDA(int i, double& lsum) {lsum += x_data_[i]*y_data_[i];}, result);
    Kokkos::fence();

    Kokkos::Timer timer;
    for (int r = 0; r < R; r++) {
      Kokkos::parallel_reduce("kk_wup", N, KOKKOS_LAMBDA(int i, double& lsum) {lsum += x_data_[i]*y_data_[i];}, result);
    }
    Kokkos::fence();
    double time = timer.seconds();
    return time;
  }

  double sycl_dot(int R) {
    DOT f(*this);
    int N_ = N;
    sycl::queue q{sycl::property::queue::in_order()};
    auto x_data_ = x_data;
    auto y_data_ = y_data;
    // Warmup
    double result = 0.;
    double* result_ptr = sycl::malloc_device<double>(1, q);
    q.submit([&](sycl::handler& cgh) {
		    cgh.parallel_for(sycl::range<1>(N_), sycl::reduction(result_ptr, 0., sycl::plus<double>()),
				    [=](sycl::id<1> idx, auto&sum) {
				    int i = idx;
				    sum += x_data_[i]*y_data_[i];
						    });});
    q.memcpy(&result, result_ptr, sizeof(double));
    q.wait();

    Kokkos::Timer timer;
    for (int r = 0; r < R; r++) {
                    q.parallel_for(sycl::range<1>(N_), sycl::reduction(result_ptr, 0., sycl::plus<double>()),
                                    [=](sycl::id<1> idx, auto&sum) {
                                    int i = idx;
				    sum += x_data_[i]*y_data_[i];});
      q.memcpy(&result, result_ptr, sizeof(double));
      q.wait();
    }
    q.wait();
    sycl::free(result_ptr, q);
    double time = timer.seconds();
    return time;
  }

  void run_test(int R) {
    double bytes_moved = 1. * sizeof(double) * N * 2 * R;
    double GB = bytes_moved / 1024 / 1024 / 1024;
    double time_kk = kk_dot(R);
    double time_sycl = sycl_dot(R);
    std::cout << "DOT KK " << N << ":\t" << time_kk << " s\t" << GB/time_kk << " GB/s" << time_sycl << " s\t" << GB/time_sycl << " GB/s\t" << time_kk/time_sycl << '\n';
  }
};
