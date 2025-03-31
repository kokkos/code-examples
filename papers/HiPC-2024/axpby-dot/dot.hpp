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
#include <Kokkos_Random.hpp>
#include <cmath>

struct DOT {
    using view_t = Kokkos::View<double*>;
    int N;
    view_t x, y;

    bool fence_all;
    DOT(int N_, bool fence_all_)
        : N(N_), x(view_t("X", N)), y(view_t("Y", N)), fence_all(fence_all_) {
        Kokkos::deep_copy(x, 1);
        Kokkos::deep_copy(y, 2);

        // Filling with random digits fails at runtime for the N=33554432
        // Kokkos::Random_XorShift64_Pool<> rand_pool64(5374857);
        // Kokkos::fill_random(x,rand_pool64,100);
        // Kokkos::fill_random(y,rand_pool64,100);
    }

    KOKKOS_FUNCTION
    void operator()(int i, double& lsum) const { lsum += x(i) * y(i); }

    double kk_dot(int R) {
        // Warmup
        double result;
        Kokkos::parallel_reduce("kk_dot_wup", N, *this, result);
        Kokkos::fence();

        Kokkos::Timer timer;
        for (int r = 0; r < R; r++) {
            Kokkos::parallel_reduce("kk_dot", N, *this, result);
        }
        Kokkos::fence();
        double time = timer.seconds();
        return time;
    }

#if defined(KOKKOS_ENABLE_OPENMPTARGET)
    double native_openmp_dot(int R) {
        DOT f(*this);
        const auto y_ = y.data();
        const auto x_ = x.data();

        // Warmup
        {
            double result = 0.;
#pragma omp target teams distribute parallel for is_device_ptr(x_, y_)\
            map(to:N) reduction(+:result)
            for (int i = 0; i < N; ++i) {
                result += x_[i] * y_[i];
            }
        }

        Kokkos::Timer timer;
        for (int r = 0; r < R; r++) {
            double result = 0.;
#pragma omp target teams distribute parallel for is_device_ptr(x_, y_) \
            map(to:N) reduction(+:result)
            for (int i = 0; i < N; ++i) {
                result += x_[i] * y_[i];
            }
        }
        double time = timer.seconds();
        return time;
    }

    double lambda_openmp_dot(int R) {
        DOT f(*this);
        const auto y_ = y.data();
        const auto x_ = x.data();

        auto dot_lambda = [=](int i, double& result) {
            result += x_[i] * y_[i];
        };

        // Warmup
        {
            double result = 0.;
#pragma omp target teams distribute parallel for is_device_ptr(x_, y_) \
      map(to:dot_lambda,N) reduction(+:result)
            for (int i = 0; i < N; ++i) {
                dot_lambda(i, result);
            }
        }

        double result = 0.;
        Kokkos::Timer timer;
        for (int r = 0; r < R; r++) {
#pragma omp target teams distribute parallel for is_device_ptr(x_, y_) \
      map(to:dot_lambda,N) reduction(+:result)
            for (int i = 0; i < N; ++i) {
                dot_lambda(i, result);
            }
        }
        double time = timer.seconds();
        result = 0;

        return time;
    }
#endif

    void run_test(int R) {
        double bytes_moved = 1. * sizeof(double) * N * 2 * R;
        double GB = bytes_moved / 1024 / 1024 / 1024;
        printf("Bytes moved[GBs] = %f\n", GB);

        // DOT as Kokkos kernels
        double time_kk = kk_dot(R);
        printf("DOT KK: %e s %e GiB/s\n", time_kk, GB / time_kk);

#if defined(KOKKOS_ENABLE_OPENMPTARGET)
        // DOT as LAMBDA inside OpenMP
        double time_lambda_openmp = lambda_openmp_dot(R);
        printf("DOT lambda-openmp: %e s %e GiB/s\n", time_lambda_openmp,
               GB / time_lambda_openmp);

        // DOT as native OpenMP
        double time_native_openmp = native_openmp_dot(R);
        printf("DOT native-openmp: %e s %e GiB/s\n", time_native_openmp,
               GB / time_native_openmp);
#endif
        printf("\n");
    }
};
