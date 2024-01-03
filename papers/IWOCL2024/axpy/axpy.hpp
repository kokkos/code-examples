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
    view_t x, y, z;

    bool fence_all;
    AXPBY(int N_, bool fence_all_)
        : N(N_),
          x(view_t("X", N)),
          y(view_t("Y", N)),
          z(view_t("Z", N)),
          fence_all(fence_all_) {}

    KOKKOS_FUNCTION
    void operator()(int i) const { for (int j=0; j<10000; ++j) z(i) += x(i) + y(i); }

      #ifdef KOKKOS_ENABLE_SYCL
  double sycl_axpby(int R) {
    AXPBY f(*this);
    int N_ = N;
    sycl::queue q{sycl::property::queue::in_order()};
    auto x_data = x.data();
    auto y_data = y.data();
    auto z_data = z.data();
    // Warmup
                    q.parallel_for(sycl::range<1>(N_), [=](sycl::id<1> idx) {
                                    int i = idx;
				    z_data[i] = x_data[i] + y_data[i];
				    });
    q.wait();

    Kokkos::Timer timer;
    for (int r = 0; r < R; r++) {
	                        q.parallel_for(sycl::range<1>(N_), [=](sycl::id<1> idx) {
                                     int i = idx;
				     for (int j=0; j<20000; ++j)
				     z_data[i] += x_data[i] + y_data[i];
                                    });
      //q.wait();
    }
    q.wait();
    double time = timer.seconds();
    return time;
  }
#endif

    double kk_axpby(int R) {
        // Warmup
        Kokkos::parallel_for("kk_axpby_wup", N, *this);
        Kokkos::fence();

        Kokkos::Timer timer;
        for (int r = 0; r < R; r++) {
            Kokkos::parallel_for("kk_axpby", N, *this);
            //Kokkos::fence();
        }
        Kokkos::fence();
        double time = timer.seconds();
        return time;
    }

#ifdef KOKKOS_ENABLE_OPENMPTARGET
    double ompt_axpby(int R) {
        AXPBY f(*this);
        int N_ = N;

// Warmup
#pragma omp target teams distribute parallel for simd map(to : f, N_)
        for (int i = 0; i < N_; i++) {
            f(i);
        }
        Kokkos::fence();

        Kokkos::Timer timer;

        for (int r = 0; r < R; r++) {
#pragma omp target teams distribute parallel for simd map(to : f, N_)
            for (int i = 0; i < N_; i++) {
                f(i);
            }
            if (fence_all) Kokkos::fence();
        }

        Kokkos::fence();
        double time = timer.seconds();
        return time;
    }

    double ompt_raw_axpby(int R) {
        double *xp = x.data();
        double *yp = y.data();
        double *zp = z.data();
        int N_ = N;

// Warmup
#pragma omp target teams distribute parallel for simd is_device_ptr( \
        xp, yp, zp) \
map(to : N_)
        for (int i = 0; i < N_; i++) {
            zp[i] = xp[i] + yp[i];
        }
        Kokkos::fence();

        Kokkos::Timer timer;
        for (int r = 0; r < R; r++) {
#pragma omp target teams distribute parallel for simd is_device_ptr( \
        xp, yp, zp) \
map(to : N_)
            for (int i = 0; i < N_; i++) {
                zp[i] = xp[i] + yp[i];
            }
            if (fence_all) Kokkos::fence();
        }
        Kokkos::fence();
        double time = timer.seconds();
        return time;
    }
#endif

    void run_test(int R) {
        double bytes_moved = 1. * sizeof(double) * N * 3 * R;
        //double GB = bytes_moved / 1024 / 1024 / 1024;
	double FLOPS = 20000.*N*R;

        // Kokkos version
        double time_kk = kk_axpby(R);
	double time_sycl = sycl_axpby(R);
	std::cout << N << ":\t" << time_kk << " s\t" << FLOPS/time_kk << " GB/s" << time_sycl << " s\t" << FLOPS/time_sycl << " GB/s\t" << time_kk/time_sycl << '\n';
#ifdef KOKKOS_ENABLE_OPENMPTARGET

        // Native OpenMPTarget version that passes a functor to the target region.
        double time_ompt = ompt_axpby(R);
        printf("AXPBY OMPT: %e s %e GB/s\n", time_ompt, GB / time_ompt);

        // Native OpenMPTarget version that uses raw pointers instead of a functor.
        double time_ompt_raw = ompt_raw_axpby(R);
        printf("AXPBY OMPT_RAW: %e s %e GB/s\n", time_ompt_raw,
               GB / time_ompt_raw);
#endif
    }
};
